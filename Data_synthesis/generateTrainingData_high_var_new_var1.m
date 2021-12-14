function generateTrainingData_high_var_new_var1

% generate a continuous 3d surface with high variance. This is for DL
% training. There is no break between two consecutive years.

close all; clear; clc;
rng(1, 'twister');


% T = 10000;
% rndWalk = GPanimation(5, T);
% mu = 1 + 10*rand(5, 1);
% std_mat = 3*rand(5, 5);
% 
% cov_mat = std_mat*std_mat';
% L = chol(cov_mat + 1e-8*eye(5));
% 
% x = zeros(5, T);
% 
% for i =1:T
%     x(:,i) = mu.*rndWalk(:,i);
% end
% 
% figure; hold on;
% for i =1:5
%     plot(x(i,:));
% end
% 
% 
% figure; hold on;
% for i =1:5
%     plot(rndWalk(i,:));
% end
% 
% xx = 1;



data = xlsread('ThreeDatasetsIowaSoils30yrsEachWithWeather.xlsx', 'Ames');
% data = xlsread('TallerFescueWithFertilizerAnnuallyIA.xlsx', '30 years');
days = data(:,1);
years = data(:,3);
heights = data(:,5);
nRows = size(heights, 1);

% vidName = 'animation.avi';
% if 2 == exist('animation.avi', 'file')
%     delete 'animation.avi';
% end
% 
% vidObj = VideoWriter(vidName);
% vidObj.FrameRate = 15;
% vidObj.Quality = 100;
% open(vidObj);

nBases = 7;
set.nBases = nBases;
bases = cell(nBases,1);
bases{1}.xi = [100;100];
bases{1}.si = 25;
bases{2}.xi = [60;80];
bases{2}.si = 25;
bases{3}.xi = [40;30];
bases{3}.si = 30;
bases{4}.xi = [160;160];
bases{4}.si = 35;
bases{5}.xi = [160;30];
bases{5}.si = 25;
bases{6}.xi = [20;20];
bases{6}.si = 25;
bases{7}.xi = [20;180];
bases{7}.si = 50;
set.bases = bases;
set.K = [5,5,3,8,4,4,5]/1.2;

SAMPLE_N = 100;
xRange = 200;
yRange = 200;
plotRange = [0 xRange 0 yRange];

M = nRows+1;
x = linspace(1, 500, M)';
kxx = kx(x,x); % kernel function (enter your favorite here)
m = zeros(M,1);
V = kxx;
L = chol(V + 1.0e-8 * eye(M)); % jitter for numerical stability

for i=1:nBases
    v = GPanimation(M,1);
    set.zz(:,i) = m + L' * v;
end
set.zz(1,:) = [];
set.zz = set.zz + abs(min(set.zz, [], [1 2]));

set.zz(:,1) = 1.2*set.zz(:,1);
set.zz(:,2) = 1.8*set.zz(:,2);
set.zz(:,3) = 1.0*set.zz(:,3);
set.zz(:,4) = 1.2*set.zz(:,4);
set.zz(:,5) = 1.5*set.zz(:,5);

if nBases == 7
    set.zz(:,6) = 1.1*set.zz(:,6);
    set.zz(:,7) = 1.4*set.zz(:,7);
end

% figure; hold on;
% for i=1:nBases
%     plot(x(2:end), set.zz(:,i));
% end
% 
% figure;  plot(set.zz(:,1));
% xlabel('Day'); 
% ylabel('The weight of one basis function');
% xlim([0 10950]);
% 
% xx = 1;

% set.zz = zeros(nRows, nBases);

% std_mat = 3*rand(nBases, nBases);
% cov_mat = std_mat*std_mat';
% L = chol(cov_mat + 1e-8*eye(nBases));
% rndWalk = GPanimation(nBases, nRows);
% mu = 1 + 5*rand(nBases, 1);

% figure; hold on;
% for i =1:5
%     plot(rndWalk(i,:));
% end


x = linspace(plotRange(1),plotRange(2), SAMPLE_N);
y = linspace(plotRange(3), plotRange(4), SAMPLE_N);
[X,Y] = meshgrid(x,y);
Z = zeros(size(X));

C = cell(size(X));

for i=1:size(X,1)
    for j=1:size(X,2)
        
        x = [X(i,j); Y(i,j)];
        cc = zeros(set.nBases,1);
        for k=1:set.nBases
            cc(k) = set.K(k)*exp(-norm(x-set.bases{k}.xi)^2/(2*set.bases{k}.si^2));
        end
        
        C{i,j} = cc;
    end
end

final = zeros(nRows, 100*100+2);

% fig = figure('Position', [200, 200, 1100, 800]);

for k=1:numel(heights)
    curYear = years(k);
    curDay = days(k);
    
    
    for i=1:size(X,1)
        for j=1:size(X,2)
            cc = C{i,j};
            Z(i,j) = cc'*set.zz(k,:)';
        end
    end
    
%     set.z = set.zz(k,:)';
%     for i=1:size(X,1)
%         for j=1:size(X,2)
%             [p,  ~] =  envSampleAtPoint(set, [X(i,j); Y(i,j)]);
%             Z(i,j) = p;
%         end
%     end
    
    Z = Z + normrnd(5,3,[SAMPLE_N,SAMPLE_N]);
    Z = Z + (heights(k) - mean(Z(:)));
    Z(Z<0) = 0;
    
    xx = 1;
    
    %     subplot(1,2,1);
%     surf(X, Y, Z, 'EdgeColor','none','FaceColor','interp'); zlim([0, 250]);
%     msg = ['year: ', num2str(curYear), ', day: ', num2str(curDay)];
%     title(msg);
%     xticks([0, 50, 100, 150, 200]);
%     xticklabels({'0', '2.5', '5', '7.5', '10'});
%     yticks([0, 50, 100, 150, 200]);
%     yticklabels({'0', '2.5', '5', '7.5', '10'});
%     xlabel('x (m)'); ylabel('y (m)'); zlabel('z (mm)');

%     xx = 1;
%     frame = getframe(fig);
%     writeVideo(vidObj, frame);
    %     subplot(1,2,2); plot(curHeights);
    %     titleName = strcat('day ', num2str(day(k)));
    %     title(titleName);
    
    %     Z1 = interp2(X,Y,Z, X1, Y1);
    
    
    final(k,:) = [curDay, curYear, Z(:)'];
    
%     set = changeBases(set, xRange);
    
%     if (min(Z(:)) < 0)
%         tmp = 1;
%     end
    
%     pause(.1);
    
    disp(k);
    
end
% close(vidObj);

% save('xx.mat', 'final');

% generate headers
header1 = cell(1, 2);
header1{1} = 'day_of_year';
header1{2} = 'year';

header2 = cell(1, 100^2);
for i=1:100^2
    header2{i} = strcat('h', num2str(i));
end

headers=[header1, header2];


if isfile('data_new_1.csv')
    delete('data_new_1.csv');
end
% xlswrite('data.csv',final);
csvwrite_with_headers('data_new_1.csv',final,headers);

% if isfile('data.csv')
%     delete('data.csv');
% end
% % xlswrite('data.csv',final);
% csvwrite_with_headers('data.csv',final,headers);

xx = 1;



end

function set = changeBases(set, xRange)

nBases = set.nBases;

for i=1:nBases
    xi = set.bases{i}.xi;
    xi = xi + .3*randn(2,1);
    xi(xi<0) = 0;
    xi(xi > xRange) = xRange;
    set.bases{i}.xi = xi;
end

end


function X = GPanimation(d,n)
% returns a matrix X of size [d,n], representing a grand circle on the
% unit d-sphere in n steps, starting at a random location. Given a kernel
% matrix K, this can be turned into a tour through the sample space, simply by
% calling chol(K)� * X;
%
% Philipp Hennig, September 2012

x = randn(d,1); % starting sample
r = sqrt(sum(x.^2));
x = x ./ r; % project onto sphere

t = randn(d,1); % sample tangent direction
t = t - (t'*x) * x; % orthogonalise by Gram-Schmidt.
t = t ./ sqrt(sum(t.^2)); % standardise
s = linspace(0,2*pi,n+1); s = s(1:end-1); % space to span
t = bsxfun(@times,s,t); % span linspace in direction of t
X = r.* exp_map(x,t); % project onto sphere, re-scale
end

function M = exp_map (mu, E)
% Computes exponential map on a sphere
%
% many thanks to Soren Hauberg!
D = size(E,1);
theta = sqrt(sum((E.^2)));
M = mu * cos(theta) + E .* repmat(sin(theta)./theta, D, 1);
if (any (abs (theta) <= 1e-7))
    for a = find (abs (theta) <= 1e-7)
        M (:, a) = mu;
    end % for
end % if
M (:,abs (theta) <= 1e-7) = mu;
end % function

function mat = kx(x1,x2)

    n = numel(x1);
    mat = zeros(n, n);
    for i=1:numel(x1)
        for j=1:numel(x2)
            mat(i,j) = 2*exp(-norm(x1(i)-x2(j))^2/(2*1^2));
        end
    end

end






































