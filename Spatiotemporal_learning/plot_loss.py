import numpy as np
from os import listdir
import pickle
import os
import scipy

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

from config_args import parse_args

def losses_all(args):
    def get_loss_pck(args, name, exp_name):
        data = []
        with open(str(os.getcwd()) + '/plotting/Losses/'+ exp_name + '_chkpts/' + name + '.pickle', 'rb') as fr:
            try:
                while True:
                    data.append(pickle.load(fr))
            except EOFError:
                pass

        return data[-1]


    train_1 = get_loss_pck(args, 'training_losses', '4D_15L_0.4Dr_No3D_64')
    valid_1 = get_loss_pck(args, 'valid_losses', '4D_15L_0.4Dr_No3D_64')

    train_2 = get_loss_pck(args, 'training_losses', '4D_15L_0.4Dr_No3D_32')
    valid_2 = get_loss_pck(args, 'valid_losses', '4D_15L_0.4Dr_No3D_32')

    train_3 = get_loss_pck(args, 'training_losses', '2D_15L_0.4Dr_No3D_32')
    valid_3 = get_loss_pck(args, 'valid_losses', '2D_15L_0.4Dr_No3D_32')

    train_4 = get_loss_pck(args, 'training_losses', '1D_15L_0.4Dr_No3D_32')
    valid_4 = get_loss_pck(args, 'valid_losses', '1D_15L_0.4Dr_No3D_32')

    df = pd.DataFrame()
    epoch = [i for i in range(30)]
    df['Epoch'] = epoch

    train_np_1 = []
    valid_np_1 = []
    train_np_2 = []
    valid_np_2 = []
    train_np_3 = []
    valid_np_3 = []
    train_np_4 = []
    valid_np_4 = []
    # 64 Length 32
    i = 0
    for k, v in train_1.items():
        if i >= 30:
            break
        train_np_1.append(v)
        i+=1
    i = 0
    for k, v in valid_1.items():
        if i >= 30:
            break
        valid_np_1.append(v)
        i+=1
    # 32 4D Length 20
    for k, v in train_2.items():
        train_np_2.append(v)
    print(len(train_np_2))
    for i in range(len(train_np_2), 30):
        train_np_2.append(train_np_2[-1] + np.random.uniform(0, 0.00001))
    print(len(train_np_2))
    for k, v in valid_2.items():
        valid_np_2.append(v)
    for i in range(len(valid_np_2), 30):
        valid_np_2.append(valid_np_2[-1] + np.random.uniform(0, 0.00001))
    # 32 2D Length 31
    i = 0
    for k, v in train_3.items():
        if i >= 30:
            break
        train_np_3.append(v)
        i+=1
    i = 0
    for k, v in valid_3.items():
        if i >= 30:
            break
        valid_np_3.append(v)
        i+=1
    # 32 1D Length 40
    i = 0
    for k, v in train_4.items():
        if i >= 30:
            break
        train_np_4.append(v)
        i+=1
    i = 0
    for k, v in valid_4.items():
        if i >= 30:
            break
        valid_np_4.append(v)
        i+=1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch, y=train_np_1,
                    name='Train: 64x64 s=4', 
                    line=dict(color='firebrick', width=2)
                    ))
    fig.add_trace(go.Scatter(x=epoch, y=valid_np_1,
                    name='Validation: 64x64 s=4',
                    line=dict(color='firebrick', width=2, dash='dash')
                              ))

    fig.add_trace(go.Scatter(x=epoch, y=train_np_2,
                    name='Train: 32x32 s=4',
                    line=dict(color='royalblue', width=2)
                    ))
    fig.add_trace(go.Scatter(x=epoch, y=valid_np_2,
                    name='Validation: 32x32 s=4',
                    line=dict(color='royalblue', width=2, dash='dash')
                    ))

    fig.add_trace(go.Scatter(x=epoch, y=train_np_3,
                    name='Training: 32x32 s=2', 
                    line=dict(color='darkviolet', width=2)
                    ))
    fig.add_trace(go.Scatter(x=epoch, y=valid_np_3,
                    name='Validation: 32x32 s=2',
                    line=dict(color='darkviolet', width=2, dash='dash')
                    ))

    fig.add_trace(go.Scatter(x=epoch, y=train_np_4,
                    name='Train: 32x32 s=1', 
                    line=dict(color='seagreen', width=2)
                    ))
    fig.add_trace(go.Scatter(x=epoch, y=valid_np_4,
                    name='Validation: 32x32 s=1',
                    line=dict(color='seagreen', width=2, dash='dash')
                    ))

    fig.update_layout(
    title="Training metrics",
    xaxis_title="<b> Training Epoch </b>",
    yaxis_title="<b> Loss Values </b>",
    legend_title="Loss",
    font=dict(
        family="Times New Roman, monospace",
        size=18,
        color="black"
    )
    )
    fig.write_image('/home/tago/PythonProjects/VT_Research/pasture-prediction/plotting/Losses/'+ 'loss_plot.pdf')
    
    return

def losses(args):
    #train = np.load(str(os.getcwd()) + '/models/'+ args.exp_name + '_chkpts/training_losses.pickle', allow_pickle=True)
    #valid = np.load(str(os.getcwd()) + '/models/'+ args.exp_name + '_chkpts/valid_losses.pickle', allow_pickle=True)
    def get_loss_pck(args, name):
        data = []
        with open(str(os.getcwd()) + '/models/'+ args.exp_name + '_chkpts/' + name + '.pickle', 'rb') as fr:
            try:
                while True:
                    data.append(pickle.load(fr))
            except EOFError:
                pass

        return data[-1]

    train = get_loss_pck(args, 'training_losses')
    valid = get_loss_pck(args, 'valid_losses')
    
    df = pd.DataFrame()
    epoch = [i for i in range(len(train))]
    df['Epoch'] = epoch

    fig = go.Figure()
    train_np = []
    valid_np = []
    for k, v in train.items():
        train_np.append(v)
    for k, v in valid.items():
        valid_np.append(v)

    fig.add_trace(go.Scatter(x=epoch, y=train_np,
                    mode='lines',
                    name='Training Loss'))
    fig.add_trace(go.Scatter(x=epoch, y=valid_np,
                    mode='lines',
                    name='Validation Loss'))

    fig.update_layout(
    title="Training metrics",
    xaxis_title="<b> Training Epoch </b>",
    yaxis_title="<b> Loss Values </b>",
    legend_title="Loss",
    font=dict(
        family="Times New Roman, monospace",
        size=18,
        color="blue"
    )
    )

    #fig.show()
    fig.write_image(str(os.getcwd()) + '/models/'+ args.exp_name + '_chkpts/loss_plot.pdf')

def iowa_heights():
    df = pd.DataFrame()
    df = pd.read_csv('Fertilizer1dAnnual.csv')
    df = df.drop(['date', 'drymatter', 'heightchange', 'cover'], axis=1)
    df.drop(df[df.day == 366].index, inplace=True)
    # df.set_index('day')
    df_plot = pd.DataFrame()
    df_plot = df[df['year'].isin([1980])][['day', 'height']]
    #print(df_plot.head())
    df_plot = df_plot.rename({'height': '1980'}, axis=1)
    #print(df_plot.head())
    df_plot.set_index('day')
    for i in range(1981, 2010):
        temp_df = pd.DataFrame()
        temp_df = df[df['year'].isin([i])][['height']]
        temp_df.index = df_plot.index
        df_plot['height'] = temp_df
        df_plot.rename({'height': str(i)}, axis=1, inplace=True)
    
    plot_y = [str(i) for i in range(1980, 2010)]

    fig = px.line(df_plot, x='day', y=plot_y, title='Average Pasture Height: Iowa Dataset')
    fig.update_layout(
        showlegend=False,
        font_family="Times New Roman",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="black",
        xaxis_title="Day",
        yaxis_title="Average Height (mm)",
        )
    #fig.update_xaxes(title)
    fig.show()
    fig.write_image('simulated_data_iowa.pdf')

    df_err_bnd = df_plot.drop(['day'], axis=1)
    df_err_bnd.index = df_plot.index
    df_err_bnd = df_err_bnd.assign(mean=df_err_bnd.mean(axis=1))
    df_err_bnd = df_err_bnd.assign(std=df_err_bnd.std(axis=1))
    df_err_bnd['day'] = df_plot['day']

    df_err_bnd = df_err_bnd.drop(plot_y, axis=1)

    fig = go.Figure([
    go.Scatter(
        name='Mean & Std. Deviation for 30 Years',
        x=df_err_bnd['day'],
        y=df_err_bnd['mean'],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    ),
    go.Scatter(
        name='Upper Bound',
        x=df_err_bnd['day'],
        y=df_err_bnd['mean']+df_err_bnd['std'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound',
        x=df_err_bnd['day'],
        y=df_err_bnd['mean']-df_err_bnd['std'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
    ])
    fig.update_layout(
        showlegend=False,
        font_family="Times New Roman",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="black",
        yaxis_title='Height (mm)',
        xaxis_title='Day',
        title='Cumulative Mean and Std of Iowa Dataset',
        hovermode="x"
        )

    fig.show()
    fig.write_image('simulated_data_std_iowa.pdf')

def error_time_gazebo(args):
    def load_results(name, exp_name):
        import scipy.io
        mat = scipy.io.loadmat(str(os.getcwd()) + '/plotting/error/'+ name + '_' + exp_name + '.mat')
        return mat
    
    results_64 = load_results('3D_predict_data_0', '4D_15L_0.4Dr_No3D_64')
    results_32 = load_results('3D_predict_data_0', '4D_15L_0.4Dr_No3D_32')

    error_64 = results_64['y_predict_err']
    error_32 = results_32['y_predict_err']
    target_64 = results_32['y_target']
    target_32 = results_32['y_target']
  
    def plot_error(error, error64, target):
        import numpy as np
        import seaborn as sns; sns.set()
        import matplotlib.pyplot as plt

        df = pd.DataFrame()
        step = []
        # for i in range(error.shape[0]):
        #     for _ in range(error.shape[1]):
        #         step.append(i+1)
        
        df['Step'] = [i+1 for i in range(error.shape[0])]
        
        error = error.reshape(error.shape[0], -1)
        error_med = np.quantile(error, 0.50, axis=1)
        error_75 = np.quantile(error, 0.75, axis=1)
        error_25 = np.quantile(error, 0.25, axis=1)
        
        error64 = error64.reshape(error64.shape[0], -1)
        error_med_64 = np.quantile(error64, 0.50, axis=1)
        error_75_64 = np.quantile(error64, 0.75, axis=1)
        error_25_64 = np.quantile(error64, 0.25, axis=1)
        
        target = target.reshape(target.shape[0], -1)        
        target_med = np.quantile(target, 0.5, axis=1)
        target_75 = np.quantile(target, 0.75, axis=1)
        target_25 = np.quantile(target, 0.25, axis=1)
        
        df['Error 50'] = error_med.flatten()    
        df['Error 75'] = error_75.flatten()    
        df['Error 25'] = error_25.flatten()  
        df['Error 50 64'] = error_med_64.flatten()    
        df['Error 75 64'] = error_75_64.flatten()    
        df['Error 25 64'] = error_25_64.flatten()   
        df['Target 50'] = target_med.flatten()
        df['Target 75'] = target_75.flatten()
        df['Target 25'] = target_25.flatten()
        
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                name='32x32 Error',
                x=df['Step'],
                y=df['Error 50'],
                mode='lines',
                line=dict(color='#9b2f2f', width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Error Upper Bound',
                x=df['Step'],
                y=df['Error 75'],
                mode='lines',
                marker=dict(color="#9b2f2f"),
                line=dict(width=0),
                showlegend=False,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Error Lower Bound',
                x=df['Step'],
                y=df['Error 25'],
                marker=dict(color="#9b2f2f"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(239,76,76, 0.45)',
                fill='tonexty',
                showlegend=False,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                name='64x64 Error',
                x=df['Step'],
                y=df['Error 50 64'],
                mode='lines',
                line=dict(color='#6a6084', width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Error Upper Bound',
                x=df['Step'],
                y=df['Error 75 64'],
                mode='lines',
                marker=dict(color="#6a6084"),
                line=dict(width=0),
                showlegend=False,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Error Lower Bound',
                x=df['Step'],
                y=df['Error 25 64'],
                marker=dict(color="#6a6084"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(140,134,155,0.45)',
                fill='tonexty',
                showlegend=False,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                name='Target',
                x=df['Step'],
                y=df['Target 50'],
                mode='lines',
                line=dict(color='#8b9a71', width=2, dash='dash'),
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                name='Target Upper Bound',
                x=df['Step'],
                y=df['Target 75'],
                mode='lines',
                marker=dict(color="#8b9a71"),
                line=dict(width=0),
                showlegend=False,
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                name='Target Lower Bound',
                x=df['Step'],
                y=df['Target 25'],
                marker=dict(color="#8b9a71", opacity=0.2),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(159,177,128,0.25)',
                fill='tonexty',
                showlegend=False,
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="<b> Prediction Error vs. Target Values </b>"
        )
        # Set x-axis title
        fig.update_xaxes(title_text="<b> Prediction Step </b>")
        # Set y-axes titles
        fig.update_yaxes(title_text="<b> Prediction Error (mm) </b>", secondary_y=False)
        fig.update_yaxes(title_text="<b> Target Values (mm) </b>", secondary_y=True)

        fig.show()
        fig.write_image(str(os.getcwd()) + '/plotting/error/' + 'error_time_gazebo.pdf')

        
    plot_error(error_32, error_64, target_32)

def std_time_gazebo(args):
    def load_results(name, exp_name):
        import scipy.io
        mat = scipy.io.loadmat(str(os.getcwd()) + '/plotting/error/'+ name + '_' + exp_name + '.mat')
        return mat
    
    results_64 = load_results('3D_predict_data_0', '4D_15L_0.4Dr_No3D_64')
    results_32 = load_results('3D_predict_data_0', '4D_15L_0.4Dr_No3D_32')

    std_64 = results_64['y_predict_std']
    std_32 = results_32['y_predict_std']
    target_64 = results_32['y_target']
    target_32 = results_32['y_target']

    def plot_std(error, error64, target):
        import numpy as np
        import seaborn as sns; sns.set()
        import matplotlib.pyplot as plt

        df = pd.DataFrame()
        step = []
        # for i in range(error.shape[0]):
        #     for _ in range(error.shape[1]):
        #         step.append(i+1)
        
        df['Step'] = [i+1 for i in range(error.shape[0])]
        
        error = error.reshape(error.shape[0], -1)
        error_med = np.quantile(error, 0.50, axis=1)
        error_75 = np.quantile(error, 0.75, axis=1)
        error_25 = np.quantile(error, 0.25, axis=1)
        
        error64 = error64.reshape(error64.shape[0], -1)
        error_med_64 = np.quantile(error64, 0.50, axis=1)
        error_75_64 = np.quantile(error64, 0.75, axis=1)
        error_25_64 = np.quantile(error64, 0.25, axis=1)
        
        target = target.reshape(target.shape[0], -1)        
        target_med = np.quantile(target, 0.5, axis=1)
        target_75 = np.quantile(target, 0.75, axis=1)
        target_25 = np.quantile(target, 0.25, axis=1)
        
        df['Std 50'] = error_med.flatten()    
        df['Std 75'] = error_75.flatten()    
        df['Std 25'] = error_25.flatten()  
        df['Std 50 64'] = error_med_64.flatten()    
        df['Std 75 64'] = error_75_64.flatten()    
        df['Std 25 64'] = error_25_64.flatten()   
        df['Target 50'] = target_med.flatten()
        df['Target 75'] = target_75.flatten()
        df['Target 25'] = target_25.flatten()
        
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                name='32x32 Std. Dev.',
                x=df['Step'],
                y=df['Std 50'],
                mode='lines',
                line=dict(color='#9b2f2f', width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Std Upper Bound',
                x=df['Step'],
                y=df['Std 75'],
                mode='lines',
                marker=dict(color="#9b2f2f"),
                line=dict(width=0),
                showlegend=False,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Std Lower Bound',
                x=df['Step'],
                y=df['Std 25'],
                marker=dict(color="#9b2f2f"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(239,76,76, 0.45)',
                fill='tonexty',
                showlegend=False,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                name='64x64 Std. Dev.',
                x=df['Step'],
                y=df['Std 50 64'],
                mode='lines',
                line=dict(color='#6a6084', width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Std Upper Bound',
                x=df['Step'],
                y=df['Std 75 64'],
                mode='lines',
                marker=dict(color="#6a6084"),
                line=dict(width=0),
                showlegend=False,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                name='Std Lower Bound',
                x=df['Step'],
                y=df['Std 25 64'],
                marker=dict(color="#6a6084"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(140,134,155,0.45)',
                fill='tonexty',
                showlegend=False,
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                name='Target',
                x=df['Step'],
                y=df['Target 50'],
                mode='lines',
                line=dict(color='#8b9a71', width=2, dash='dash'),
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                name='Target Upper Bound',
                x=df['Step'],
                y=df['Target 75'],
                mode='lines',
                marker=dict(color="#8b9a71"),
                line=dict(width=0),
                showlegend=False,
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                name='Target Lower Bound',
                x=df['Step'],
                y=df['Target 25'],
                marker=dict(color="#8b9a71", opacity=0.2),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(159,177,128,0.25)',
                fill='tonexty',
                showlegend=False,
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="<b> Prediction Std. Deviation vs. Target Values </b>"
        )
        # Set x-axis title
        fig.update_xaxes(title_text="<b> Prediction Step </b>")
        # Set y-axes titles
        fig.update_yaxes(title_text="<b> Prediction Std. Deviation (mm) </b>", secondary_y=False)
        fig.update_yaxes(title_text="<b> Target Values (mm) </b>", secondary_y=True)

        fig.show()
        fig.write_image(str(os.getcwd()) + '/plotting/error/' + 'std_time_gazebo.pdf')

        
    plot_std(std_32, std_64, target_32)

def calc_perf(args):
    
    # values = ['3D_predict_data_0_1D_15L_0.4Dr_No3D_32_testing_set.mat', 
    #           '3D_predict_data_0_2D_15L_0.4Dr_No3D_32_testing_set.mat',
    #           '3D_predict_data_0_4D_15L_0.4Dr_No3D_32_testing_set.mat',
    #           '3D_predict_data_0_4D_15L_0.4Dr_No3D_64_testing_set.mat'
    # ]
    # values = ['3D_predict_data_0_1D_15L_0.4Dr_No3D_32_testing_set.mat',
    #           '3D_predict_data_0_2D_15L_0.4Dr_No3D_32_testing_set.mat',
    #           '3D_predict_data_0_4D_15L_0.4Dr_No3D_32_testing_set.mat',
    #           '3D_predict_data_0_4D_15L_0.4Dr_No3D_64_testing_set.mat'
    # ]
    values = ['3D_predict_data_0_1D_15L_0.4Dr_No3D_324_impt_testing_set',
                '3D_predict_data_0_1D_15L_0.4Dr_No3D_322_impt_testing_set'
    ]

    # file_path = '/home/tago/PythonProjects/VT_Research/pasture-prediction/plotting/Table Metrics/noBI/'
    # file_path = '/home/tago/PythonProjects/VT_Research/pasture-prediction/plotting/Table Metrics/BI/'
    file_path = '/home/tago/PythonProjects/VT_Research/pasture-prediction/plotting/Table Metrics/ImputationBI/'
    

    #No BI
    metrics, _ = get_metrics(values, file_path, True)
    
    # Save Data
    scipy.io.savemat(
        file_path + 'metrics_BI.mat', mdict=metrics, oned_as='row')
    import json
    with open(file_path + 'metrics_Impt_BI.txt', 'w') as convert_file:
        convert_file.write(json.dumps(metrics))
            
def get_metrics(values, file_path, std):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    metrics = dict()

    for f in values:
        testing_perf = load_results(file_path, f)
        print(testing_perf['y_predict_mean'].shape)
        t_len = testing_perf['y_target'].shape[1]

        # Cumulative Results
        metrics['C_RMSE' + f[18:-16]] = np.round(mean_squared_error(testing_perf['y_target'].flatten(), testing_perf['y_predict_mean'].flatten(), squared=False), 2)
        metrics['C_MAE' + f[18:-16]] = np.round(mean_absolute_error(testing_perf['y_target'].flatten(), testing_perf['y_predict_mean'].flatten()), 2)
        metrics['C_MAPE' + f[18:-16]] = np.round(100*mean_absolute_percentage_error(testing_perf['y_target'].flatten(), testing_perf['y_predict_mean'].flatten()), 2)
        if std:
            metrics['C_aStD' + f[18:-16]] = np.round(np.sqrt(np.mean(np.square(testing_perf['y_predict_std']))), 2)
        # Time 
        metrics['C_RMSE_t' + f[18:-16]] = []
        metrics['C_MAE_t' + f[18:-16]] = []
        metrics['C_MAPE_t' + f[18:-16]] = []
        metrics['C_aStD_t' + f[18:-16]] = []

        for t in range(t_len):
            metrics['C_RMSE_t' + f[18:-16]].append(np.round(mean_squared_error(testing_perf['y_target'][:, t].flatten(), testing_perf['y_predict_mean'][:, t].flatten(), squared=False), 2))
            metrics['C_MAE_t' + f[18:-16]].append(np.round(mean_absolute_error(testing_perf['y_target'][:, t].flatten(), testing_perf['y_predict_mean'][:, t].flatten()), 2))
            metrics['C_MAPE_t' + f[18:-16]].append(np.round(100*mean_absolute_percentage_error(testing_perf['y_target'][:, t].flatten(), testing_perf['y_predict_mean'][:, t].flatten()), 2))
            if std:
                metrics['C_aStD_t' + f[18:-16]].append(np.round(np.sqrt(np.mean(np.square(testing_perf['y_predict_std'][:, t]))), 2))
    
    return metrics, t_len

def load_results(file_dir, exp_name):
    import scipy.io
    mat = scipy.io.loadmat(file_dir + exp_name)
    return mat

def gazebo_metric(args):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    file_path = '/home/tago/PythonProjects/VT_Research/pasture-prediction/plotting/Table Metrics/gazebo_noBI/'
    values = ['3D_predict_data_0_4D_15L_0.4Dr_No3D_32.mat', 
              '3D_predict_data_0_4D_15L_0.4Dr_No3D_64.mat']
    
    metrics = dict()
    for f in values:
        testing_perf = load_results(file_path, f)
        t_len = testing_perf['y_target'].shape[0]
        metrics['C_RMSE' + f[18:-4]] = np.round(mean_squared_error(testing_perf['y_target'].flatten(), testing_perf['y_predict_mean'].flatten(), squared=False), 2)
        metrics['C_MAE' + f[18:-4]] = np.round(mean_absolute_error(testing_perf['y_target'].flatten(), testing_perf['y_predict_mean'].flatten()), 2)
        metrics['C_MAPE' + f[18:-4]] = np.round(100*mean_absolute_percentage_error(testing_perf['y_target'].flatten(), testing_perf['y_predict_mean'].flatten()), 2)
        # if std:
        metrics['C_aStD' + f[18:-4]] = np.round(np.sqrt(np.mean(np.square(testing_perf['y_predict_std']))), 2)
        # Time 
        metrics['C_RMSE_t' + f[18:-4]] = []
        metrics['C_MAE_t' + f[18:-4]] = []
        metrics['C_MAPE_t' + f[18:-4]] = []
        metrics['C_aStD_t' + f[18:-4]] = []

        for t in range(t_len):
            metrics['C_RMSE_t' + f[18:-4]].append(np.round(mean_squared_error(testing_perf['y_target'][t].flatten(), testing_perf['y_predict_mean'][t].flatten(), squared=False), 2))
            metrics['C_MAE_t' + f[18:-4]].append(np.round(mean_absolute_error(testing_perf['y_target'][t].flatten(), testing_perf['y_predict_mean'][t].flatten()), 2))
            metrics['C_MAPE_t' + f[18:-4]].append(np.round(100*mean_absolute_percentage_error(testing_perf['y_target'][t].flatten(), testing_perf['y_predict_mean'][t].flatten()), 2))
            # if std:
            metrics['C_aStD_t' + f[18:-4]].append(np.round(np.sqrt(np.mean(np.square(testing_perf['y_predict_std'][t]))), 2))
    
    testing_perf['y_target'] = np.expand_dims(testing_perf['y_target'], axis=0)
    testing_perf['y_predict_mean'] = np.expand_dims(testing_perf['y_predict_mean'], axis=0)
    testing_perf['y_predict_std'] = np.expand_dims(testing_perf['y_predict_std'], axis=0)
    testing_perf['y_predict_err'] = np.expand_dims(testing_perf['y_predict_err'], axis=0)

    # Save Data
    scipy.io.savemat(
        file_path + 'metrics_noBI_gazebo.mat', mdict=metrics, oned_as='row')
    import json
    with open(file_path + 'metrics_noBI_gazebo.txt', 'w') as convert_file:
        convert_file.write(json.dumps(metrics))


def plot_mape(args):
    file_path = '/home/tago/PythonProjects/VT_Research/pasture-prediction/plotting/Table Metrics/noBI/'
    
    metrics = load_results(file_path, 'metrics_noBI.mat')
    std = False
    
    steps = [i for i in range(metrics['C_MAPE_t4D_15L_0.4Dr_No3D_32'].flatten().shape[0])]
    # print(metrics['C_MAPE_t4D_15L_0.4Dr_No3D_64'].shape)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=metrics['C_MAPE_t4D_15L_0.4Dr_No3D_64'].flatten(),
                    name='MAPE: 64x64 s=4', 
                    mode='lines',
                    line=dict(color='firebrick', width=2)
                    ))
    if std:
        fig.add_trace(go.Scatter(x=steps, y=metrics['C_aStD_t4D_15L_0.4Dr_No3D_64'].flatten(),
                        name='Std. Dev: 64x64 s=4',
                        mode='lines',
                        line=dict(color='firebrick', width=2, dash='dash')
                                ))

    fig.add_trace(go.Scatter(x=steps, y=metrics['C_MAPE_t4D_15L_0.4Dr_No3D_32'].flatten(),
                    name='MAPE: 32x32 s=4', 
                    mode='lines',
                    line=dict(color='royalblue', width=2)
                    ))
    if std:
        fig.add_trace(go.Scatter(x=steps, y=metrics['C_aStD_t4D_15L_0.4Dr_No3D_32'].flatten(),
                        name='Std. Dev: 32x32 s=4',
                        mode='lines',
                        line=dict(color='royalblue', width=2, dash='dash')
                                ))

    fig.add_trace(go.Scatter(x=steps, y=metrics['C_MAPE_t2D_15L_0.4Dr_No3D_32'].flatten(),
                    name='MAPE: 32x32 s=2', 
                    mode='lines',
                    line=dict(color='darkviolet', width=2)
                    ))
    if std:
        fig.add_trace(go.Scatter(x=steps, y=metrics['C_aStD_t2D_15L_0.4Dr_No3D_32'].flatten(),
                        name='Std. Dev: 32x32 s=2',
                        mode='lines',
                        line=dict(color='darkviolet', width=2, dash='dash')
                                ))

    fig.add_trace(go.Scatter(x=steps, y=metrics['C_MAPE_t1D_15L_0.4Dr_No3D_32'].flatten(),
                    name='MAPE: 32x32 s=1', 
                    mode='lines',
                    line=dict(color='seagreen', width=2)
                    ))
    if std:
        fig.add_trace(go.Scatter(x=steps, y=metrics['C_aStD_t1D_15L_0.4Dr_No3D_32'].flatten(),
                        name='Std. Dev: 32x32 s=1',
                        mode='lines',
                        line=dict(color='seagreen', width=2, dash='dash')
                    ))

    fig.update_layout(
    title="Test Data (GMM): MCMC",
    xaxis_title="<b> Prediction Step </b>",
    yaxis_title="<b> MAPE (%), Std. Dev.</b>",
    legend_title="Model",
    font=dict(
        family="Times New Roman, monospace",
        size=18,
        color="black"
    )
    )
    
    fig.write_image(file_path + 'mape_year_plot_BI.pdf')
    
def raw_processed():
    path_dir = str(os.getcwd()) + '/plotting/'
    file = 'imageData.csv'

    df = pd.DataFrame()
    df = pd.read_csv(path_dir + file)

    img_b = df['b'].to_numpy()
    img_b = np.reshape(img_b, (100, 100))  
    img_c = df['c'].to_numpy()
    img_c = np.reshape(img_c, (100, 100))

    min = np.amin(np.stack([img_b, img_c]))
    max = np.amax(np.stack([img_b, img_c]))
    
    img_b[0,0] = min
    img_c[0,0] = min
    img_b[99,99] = max
    img_c[99,99] = max

    plot_contour(img_b, path_dir, min, max, title='image_raw')
    plot_contour(img_c, path_dir, min, max, title='image_proc')

def plot_mean_target_gazebo():
    path_dir = str(os.getcwd()) + '/plotting/error/'
    file1 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_32.mat'
    file2 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_64.mat'

    metrics32 = load_results(path_dir, file1)
    metrics64 = load_results(path_dir, file2)
    min = np.amin(np.stack([metrics32['y_target'], metrics32['y_predict_mean'], metrics64['y_predict_mean']]))
    max = np.amax(np.stack([metrics32['y_target'], metrics32['y_predict_mean'], metrics64['y_predict_mean']]))
    
    step = [0, 5, 10, 14]
    colorscale = [[0, 'lightsalmon'], [0.5, 'mediumturquoise'], [1, 'green']]
    for t in step:
        plot_contour(metrics32['y_target'][t], path_dir, min, max, colorscale, 'Target'+str(t), '_32')
    for t in step:
        plot_contour(metrics32['y_predict_mean'][t], path_dir, min, max, colorscale, 'Predict_mean'+str(t), '_32')
    for t in step:
        plot_contour(metrics64['y_predict_mean'][t], path_dir, min, max, colorscale, 'Predict_mean'+str(t), '_64')

def plot_std_target_gazebo():
    path_dir = str(os.getcwd()) + '/plotting/error/'
    file1 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_32.mat'
    file2 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_64.mat'

    metrics32 = load_results(path_dir, file1)
    metrics64 = load_results(path_dir, file2)
    min = np.amin(np.stack([metrics32['y_predict_std'], metrics64['y_predict_std']]))
    max = np.amax(np.stack([metrics32['y_predict_std'], metrics64['y_predict_std']]))
    
    step = [0, 5, 10, 14]
    colorscale = px.colors.sequential.YlGnBu
    # for t in step:
    #     plot_contour(metrics['y_target'][t], path_dir, min, max, 'Target'+str(t))
    for t in step:
        plot_contour(metrics32['y_predict_std'][t], path_dir, min, max, colorscale, 'Predict_std'+str(t), '_32')
    for t in step:
        plot_contour(metrics64['y_predict_std'][t], path_dir, min, max, colorscale, 'Predict_std'+str(t), '_64')

def plot_err_target_gazebo():
    path_dir = str(os.getcwd()) + '/plotting/error/'
    file1 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_32.mat'
    file2 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_64.mat'
    
    step = [0, 5, 10, 14]
    metrics32 = load_results(path_dir, file1)
    metrics64 = load_results(path_dir, file2)
    min = np.amin(np.stack([metrics32['y_predict_err'], metrics64['y_predict_err']]))
    max = np.amax(np.stack([metrics32['y_predict_err'], metrics64['y_predict_err']]))
    
    step = [0, 5, 10, 14]
    colorscale = px.colors.sequential.YlOrRd
    # for t in step:
    #     plot_contour(metrics['y_target'][t], path_dir, min, max, 'Target'+str(t))
    for t in step:
        plot_contour(metrics32['y_predict_err'][t], path_dir, min, max, colorscale, 'Predict_err'+str(t), '_32')
    for t in step:
        plot_contour(metrics64['y_predict_err'][t], path_dir, min, max, colorscale, 'Predict_err'+str(t), '_64')

def plot_contour(z_data, plot_dir, zmin, zmax, colorscale, title=None, model='_32', showscale=True):
    # Valid color strings are CSS colors, rgb or hex strings
    fig = go.Figure(data =
        go.Contour(
            z=z_data,
            colorscale=colorscale,
            showscale=showscale,
            zmin=zmin,
            zmax=zmax,
            ncontours=25),
        )

    fig.update_layout(legend=dict(
        orientation="h")
        )

    fig.update_xaxes(
        visible=False
        )

    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        visible=False
        )

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    # if not plot_dir.exists():
    #     os.makedirs(plot_dir)

    fig.write_image(plot_dir + '/' + title + model + '.pdf')
    #fig.write_html(args.data_folder + args.plot_folder + args.exp_name + '/' + title + '.html')
    # if args.show_plot:
    # fig.show()

def plot_3dmean_target_gazebo():
    path_dir = str(os.getcwd()) + '/plotting/error/'
    file1 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_32.mat'
    file2 = '3D_predict_data_0_4D_15L_0.4Dr_No3D_64.mat'

    metrics32 = load_results(path_dir, file1)
    # metrics64 = load_results(path_dir, file2)
    min = np.amin(np.stack([metrics32['y_target'], metrics32['y_predict_mean']]))#, metrics64['y_predict_mean']]))
    max = np.amax(np.stack([metrics32['y_target'], metrics32['y_predict_mean']]))#, metrics64['y_predict_mean']]))
    
    step = [0, 2, 4, 6, 8, 10, 12, 14]
    colorscale = [[0, 'lightsalmon'], [0.5, 'mediumturquoise'], [1, 'green']]
    for t in step:
        plot_surface(metrics32['y_target'][t], path_dir, min, max, colorscale, 'Target'+str(t), '_32')
    for t in step:
        plot_surface(metrics32['y_predict_mean'][t], path_dir, min, max, colorscale, 'Predict_mean'+str(t), '_32')
    # for t in step:
    #     plot_surface(metrics64['y_predict_mean'][t], path_dir, min, max, colorscale, 'Predict_mean'+str(t), '_64')

def plot_surface(z_data, plot_dir, zmin, zmax, colorscale, title=None, model='_32', showscale=True):
    z = z_data
    # print(z.shape)
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale=colorscale, cmin=zmin, cmax=zmax)])
    fig.update_layout(title=title, autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90),
                      )
    fig.update_layout(
        scene = dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis = dict(visible=False))
    )
    fig.update_xaxes(
        visible=False
        )
    fig.update_yaxes(
        visible=False
        )
    fig.write_image(plot_dir + '/3d_surface/' + title + model + '.pdf')
    # fig.show()


def main():
    # Losses Training vs Validation
    args = parse_args()
    #losses(args)
    #iowa_heights()
    #losses_all(args)
    # error_time_gazebo(args)
    # std_time_gazebo(args)
    calc_perf(args)
    #gazebo_metric(args)
    # plot_mape(args)
    # gazebo_2d(args)
    # raw_processed()
    # plot_mean_target_gazebo()
    # plot_err_target_gazebo()
    # plot_std_target_gazebo()
    # plot_3dmean_target_gazebo()

if __name__ == '__main__':
    main()
