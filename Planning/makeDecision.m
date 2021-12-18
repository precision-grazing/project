function [globalIdxes, policies, vals] = makeDecision

% FIGURES.

close all; clear; clc;
%     rng('default');

dayLimit = 3;
perDayLimit = 4;

isSave = 0;
isPlot = 1;

nDays = 15;

totalLimit = dayLimit*perDayLimit;

w0 = 2;
w1 = 3; % time penalty
w2 = 5*fix(nDays /(10*sqrt(2))); % dist function: dist
w3 = 1; % dist function: time

% w1 = 1; % dist: location
% w2 = fix(nDays /(10*sqrt(2))); % dist: time
% w3 = 5; 

policies = cell(3,1); % x, y, t
globalIdxes = cell(3,1); % idx
objVals = zeros(3,1);

% stepSize = 2;
xDim = 100;
yDim = 100;
dim = xDim*yDim;

x = linspace(1, xDim, xDim);
y = linspace(1, yDim, yDim);
[xx,yy] = meshgrid(x,y);

mus = [];
stds = [];
errors = [];
groundTruths = [];
dayIdx = [];

for i=1:nDays
    cur = squeeze(y_predict_mean(i,:,:));
    mus = [mus; cur(:)];

    cur = squeeze(y_predict_std(i,:,:));
    stds = [stds; cur(:)];
    
    cur = squeeze(y_predict_err(i,:,:));
    errors = [errors; cur(:)];
    
    cur = squeeze(y_target(i,:,:));
    groundTruths = [groundTruths; cur(:)];
    
    dayIdx = [dayIdx; i*ones(dim, 1)];
end

allLocations = [xx(:), yy(:)];
groundSet = [repmat(allLocations, nDays,1), dayIdx]; % 3 by 1: [x, y, day idx].


% ===================== random.
days = randperm(nDays);
sampledDays = days(1:dayLimit);
sampledDays = sort(sampledDays);

policy = [];
globalIdx = [];

for i=sampledDays
    curIdx = randperm(dim)';
    curIdx = curIdx(1:perDayLimit, :);
    policy = [policy; curIdx, i*ones(perDayLimit, 1)]; % policy is local
    
    idx1 = (i-1)*dim;
    globalIdx = [globalIdx; curIdx + idx1];
end

policy = [allLocations(policy(:,1), :), policy(:, 2)]; % policy: location, day index
plotResults(policy, sampledDays, y_predict_std, 'random', isPlot, isSave);
globalIdxes{1} = globalIdx;
policies{1} = policy;


% ===================== heuristic. 4 samples: 1, 34, 67, 100
% days = randperm(nDays);
% sampledDays = days(1:dayLimit);
% sampledDays = sort(sampledDays);

days = fix(linspace(1, nDays, perDayLimit));

policy = [];
globalIdx = [];
idx = linspace(1, xDim, fix(sqrt(perDayLimit)) + 2)';
idx = fix(idx);
idx = idx(2:end-1);
[X,Y] = meshgrid(idx, idx);
locations = [X(:) Y(:)];
idx = [];

for i=1:size(locations, 1)
    [~,I]=ismember(locations(i,:), allLocations, 'rows');
    idx = [idx; I];
end

for i=sampledDays
    policy = [policy; locations, i*ones(perDayLimit, 1)]; % policy is local
    
    idx1 = (i-1)*dim;
    globalIdx = [globalIdx;  idx + idx1];
end

plotResults(policy, sampledDays, y_predict_std, 'heuristic', isPlot, isSave);
globalIdxes{2} = globalIdx;
policies{2} = policy;

% days = randperm(nDays);
% sampledDays = days(1:dayLimit);
% sampledDays = sort(sampledDays);

% days = fix(linspace(1, nDays, perDayLimit));
% 
% policy = [];
% globalIdx = [];
% idx = linspace(1, xDim, fix(sqrt(perDayLimit)) + 2)';
% idx = fix(idx);
% idx = idx(2:end-1);
% [X,Y] = meshgrid(idx, idx);
% locations = [X(:) Y(:)];
% idx = [];
% 
% for i=1:size(locations, 1)
%     [~,I]=ismember(locations(i,:), allLocations, 'rows');
%     idx = [idx; I];
% end
% 
% for i=sampledDays
%     policy = [policy; locations, i*ones(perDayLimit, 1)]; % policy is local
%     
%     idx1 = (i-1)*dim;
%     globalIdx = [globalIdx;  idx + idx1];
% end
% 
% plotResults(policy, sampledDays, y_predict_std, 'heuristic', isPlot, isSave);
% globalIdxes{2} = globalIdx;
% policies{2} = policy;



% ===================== intermittent
perDayCount = zeros(nDays, 1);
totalCount = 0;
baseIdx = [1:nDays*dim]';
skippedIdx = [];
globalIdx = [];
days = [];
uniqueDays = [];

for i=1:totalLimit
    if 1 == i
        [~, idx] = max(stds);
        realIdx = idx;
    else
        usefulIdx = setdiff(baseIdx, skippedIdx);

        selected = groundSet(globalIdx, :);
        candidates = groundSet(usefulIdx, :);
        subStds = stds(usefulIdx);
        subErrors = errors(usefulIdx);
        score = scoreFunc1(selected, candidates, subStds, subErrors, w0, w1, w2, w3); 

        [~, idx] = max(score);
        realIdx = usefulIdx(idx(1));
    end

    globalIdx = [globalIdx; realIdx];
    skippedIdx = [skippedIdx; realIdx];
    day = dayIdx(realIdx);
    days = [days; day];
    uniqueDays = unique(days);
    
    % set limit
    perDayCount(day) = perDayCount(day) + 1;
    totalCount = totalCount + 1;

    if perDayCount(day) == perDayLimit
        idx1 = (day - 1)*dim+1;
        idx2 = day * dim;
        cur = idx1:idx2;
        skippedIdx = [skippedIdx; cur'];
    end
    
    if numel(uniqueDays) == dayLimit
        for j=1:nDays
            if ~ismember(j,uniqueDays)
                idx1 = (j - 1)*dim+1;
                idx2 = j * dim;
                cur = idx1:idx2;
                skippedIdx = [skippedIdx; cur'];
            end
        end
    end
    
    skippedIdx = unique(skippedIdx);
end

globalIdx = sort(globalIdx);
policy = groundSet(globalIdx, :);
plotResults(policy, uniqueDays', y_predict_std, 'intermittent', isPlot, isSave);
globalIdxes{3} = globalIdx;
policies{3} = policy;

xx = 1;

% ===================== compare results
vals = zeros(3,2);

for i=1:3
    globalIdx = globalIdxes{i};
    vals(i, :) = calObjVal(globalIdx, groundSet, stds, errors, w0, w1, w2, w3);
end

xx = 1;


% % plot result
% for i=1:numel(uniqueDays)
%     day = uniqueDays(i);
%     xlabelName = strcat("Day ", num2str(day));
%     fileName = strcat('intermittent', num2str(i),'.pdf');
%     
%     idx1 = (day - 1)*dim+1;
%     idx2 = day * dim;
%     
%     policyLocations = groundSet(globalIdx(globalIdx >= idx1 & globalIdx <= idx2), 1:2);
%     
%     curStds = squeeze(y_predict_std(uniqueDays(i),:,:));
%     plotContourf(curStds, policyLocations, xlabelName, [0,5], isSave, fileName);
% end


% ===================== compare objective function values
% collected = zeros(3,1);
% 
% for i=1:3
%     collected(i) = calObjVal(globalIdx, y_predict_std);
% end


% ===================== compare measurement vs ground truth, not useful
% diffs = zeros(3,1);
% 
% for i=1:3
%     cur = globalIdxes{i};
%     diffs(i) = sum(mus(cur) - groundTruths(cur));
% end
% 
% % diffs = diffs + abs(min(diffs));
% % diffs = abs(diffs);
% 
% figure;
% plot(diffs);
% 
% xx = 1;

end



% ==============================================================================================================================
% ==============================================================================================================================




% ===================== calculate objective function value
function val = calObjVal(globalIdx, groundSet, stds, errors, w0, w1, w2, w3)
% policy: global policy
% m = numel(globalIdx);
% vals = zeros(m,1);
% 
% scores = zeros(m,m);
% 
% for i=1:m
%     
%     selectedIdx = globalIdx(i);
%     restIdx = setdiff(globalIdx, selectedIdx);
%     
%     selected = groundSet(selectedIdx, :);
%     rest = groundSet(restIdx, :);
%     restStds = stds(restIdx);
%     vals(:,i) = scoreFunc1(selected, rest, restStds, w1, w2, w3);
% end
% 
% val = sum(vals);

m = numel(globalIdx);
scores = zeros(m,1);
itermizedScore = zeros(m,2);

for i=1:m
    
    curIdx = globalIdx(i);
    restIdx = setdiff(globalIdx, curIdx);
    
    selecteds = groundSet(restIdx, :);
    candidate = groundSet(curIdx, :);
    subStd = stds(curIdx);
    subError = errors(curIdx);
        
    [~, itermizedScore(i,:)] = scoreFunc1(selecteds, candidate, subStd, subError, w0, w1, w2, w3);
end

% val = sum(itermizedScore, 'all');
val = sum(itermizedScore);

xx = 1;

end


% ===================== calculate distance 
function dist = calDist(a, b, w2, w3)

% x = b - a;
% x1 = x(1)/10; % from 0~100 (pixel) to 0~10 (physical)
% x2 = x(2)/10;
% x3 = x(3);
% dist = w1*log(sqrt(x1^2 + x2^2)) + w2*abs(x3);
% dist = dist/2;

x = a - b;
x1 = x(:,1)/10; % from 0~100 (pixel) to 0~10 (physical)
x2 = x(:,2)/10;
x3 = x(:,3);
dist = w2*log(sqrt(x1.^2 + x2.^2) + 0.0001) + w3*abs(x3);
dist = dist/2;

end


% ===================== score function 1
function [score, itemizedScore] = scoreFunc1(selected, candidates, subStds, subErrors, w0, w1, w2, w3)

m = size(candidates, 1); % # of rows
n = size(selected, 1); % # of cols
dist = zeros(m, n);

% for i=1:n
%     for j=1:m
%         dist(j, i) = calDist(selected(i,:), candidates(j,:), w1, w2);
%     end
% end

for i=1:n
    dist(:,i) = calDist(selected(i,:), candidates, w2, w3);
end

% dist = dist/2;
% dist0 = dist;
% dist = exp(-dist);
% dist = 1./dist;
% dist = sum(dist, 2) + 1;
% score = subStds./dist - w3*selected(i, 3);

itemizedScore = zeros(m,2);
itemizedScore(:,1) = (subStds + w0).*sum(dist,2)/n;
itemizedScore(:,2) = - w1*candidates(:, 3);

score = sum(itemizedScore,2);
% score = subStds.*sum(dist,2)/n- w1*candidates(:, 3);

xx = 1;

end


% ===================== score function 2
function score = scoreFunc2(selected, candidates, subStds, w1, w2, w3)

n = size(selected, 1);
m = size(candidates, 1);
dist = zeros(m, n);

for i=1:n
    for j=1:m
        dist(j, i) = calDist(selected(i,:), candidates(j,:), w1, w2) - w3*selected(i, 3);
    end
end

dist = sum(dist, 2);
dist = exp(-dist) + 1;
score = subStds./dist;

end


% ===================== score function 3
function score = scoreFunc3(selected, candidates, subStds, w1, w2, w3)

n = size(selected, 1);
m = size(candidates, 1);
dist = zeros(m, n);

for i=1:n
    for j=1:m
        dist(j, i) = calDist(selected(i,:), candidates(j,:), w1, w2) - w3*selected(i, 3);
    end
end

dist = sum(exp(-dist), 2) + 1;
dist = subStds./dist;
score = sum(dist, 2);

end


% ===================== score function 4
function score = scoreFunc4(selected, candidates, subStds, w1, w2, w3)

n = size(selected, 1);
m = size(candidates, 1);
dist = zeros(m, n);

for i=1:n
    for j=1:m
        dist(j, i) = calDist(selected(i,:), candidates(j,:), w1, w2) - w3*selected(i, 3);
    end
end

dist = sum(2*log(dist), 2) ;
score = subStds.*dist;

end


% ===================== plot all results for the current policy
function plotResults(policy, days, y_predict_std, prefix, isPlot, isSave)

if 0 == isPlot
    return;
end

idx = 1;
for i=days
    xlabelName = strcat("Day ", num2str(days(idx)));
    fileName = strcat(prefix, num2str(idx), '.pdf');
    poss = policy(policy(:, 3) == days(idx), 1:2);
    Z = squeeze(y_predict_std(days(idx),:,:));
    
    figure; hold on;
    colorBarRange = [5 12];

    [~, fig] = contourf(Z);  % 'LineStyle', '--'
    set(fig,'LineColor','none');
%     colormap(summer);

    cb = colorbar;
%     caxis([5, 12]);
    % cb.Label.String = 'Variance (mm)';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 12;
    set(cb,'TickLabelInterpreter','latex');

    % plot(curPolicy(:, 1), curPolicy(:, 2), '.b', 'MarkerSize',2);

    

    xlabel(xlabelName);

    xticks([1, 25, 50, 75, 100]);
    xticklabels({'1', '2.5', '5', '7.5', '10'});
    yticks([1, 25, 50, 75, 100]);
    yticklabels({'1', '2.5', '5', '7.5', '10'});

%     color = [45 149 191]/255;
    color = [255 94 105]/255;
%     color = [0 204 204]/255;
    circles(poss(:,1), poss(:,2), 3, 'color', color, 'EdgeColor', color, 'FaceAlpha', .85);

%     hold off; 
    axis equal;
    dim = 100;
    xlim([1, dim]);
    ylim([1, dim]);

    if 1 == isSave
        saveaspdf(fileName);
    end
    
%     plotContourf(Z, pos, xlabelName, [0,5], isSave, fileName);
    
    idx = idx + 1;
end

end


% % ===================== plot a figure
% % function plotContourf(Z, poss, xlabelName, colorBarRange, isSave, fileName)
% % 
% % figure; hold on;
% % colorBarRange = [5 12];
% % 
% % [~, fig] = contourf(Z);  % 'LineStyle', '--'
% % set(fig,'LineColor','none');
% % colormap(summer);
% % colormap(summer);
% % 
% % cb = colorbar;
% % caxis(colorBarRange);
% % % cb.Label.String = 'Variance (mm)';
% % cb.Label.Interpreter = 'latex';
% % cb.Label.FontSize = 12;
% % set(cb,'TickLabelInterpreter','latex');
% % 
% % % plot(curPolicy(:, 1), curPolicy(:, 2), '.b', 'MarkerSize',2);
% % 
% % dim = 100;
% % xlim([1, dim]);
% % ylim([1, dim]);
% % 
% % xlabel(xlabelName);
% % 
% % xticks([1, 25, 50, 75, 100]);
% % xticklabels({'1', '2.5', '5', '7.5', '10'});
% % yticks([1, 25, 50, 75, 100]);
% % yticklabels({'1', '2.5', '5', '7.5', '10'});
% % 
% % color = [45 149 191]/255;
% % color = [0 204 204]/255;
% % 
% % for i=1:size(poss, 1)
% % %     viscircles([policyLocations(i, 1), policyLocations(i, 2)], 2,'color','b', 'LineWidth', 1);
% % %     plot( pos(:,1) + pos(:,2)*sqrt(-1),'o','LineWidth',1.5,'MarkerSize',10, 'MarkerEdgeColor', color, 'MarkerFaceColor', color);
% %     circles(poss(:,1), poss(:,2), 3, 'color', color, 'EdgeColor', color, 'FaceAlpha', .85);
% % end
% % 
% % hold off; 
% % axis equal;
% % 
% % if 1 == isSave
% %     saveaspdf(fileName);
% % end
% % 
% % end


% % % global index to local index
% % function idx2 = globalIdx2LocalIdxAndDay(idx1)
% %
% % idx2 = [];
% %
% % for i=1:size(idx1, 1)
% %     cur  = idx1(i);
% %     curIdx = mod(cur, 10000);
% %     day = floor(cur/10000) + 1;
% %     idx2 = [idx2; curIdx, day];
% % end
% %
% % end



% % global index to day
% function day = idx2day(idx)
%
% day = floor(idx/10000) + 1;
%
% end


% % % global index to location and day
% % function locations = localIdx2Location(idx, allLocations)
% %
% % locations = [];
% %
% %
% % for i=1:size(idx, 1)
% %     locations = [locations; allLocations(idx(i), :)];
% % end
% %
% % end




%
% for i=1:totalLimit
%     for j=1:nRows
%
%         if ismember(stds(j,2), skippedDays)
%             continue;
%         end
%
%         if 1 == totalCount
%             idx = stdsIdx(j);
%             day = idx2day(idx);
%         else
%
%         end
%
%         policy = [policy; idx];
%         perDayCount(day) = perDayCount(day) + 1;
%         totalCount = totalCount + 1;
%
%
%
%         if perDayCount(day) == perDayLimit
%             skippedDays = [skippedDays; day];
%         end
%
%         if totalCount == totalLimit
%             break;
%         end
%     end
% end

































