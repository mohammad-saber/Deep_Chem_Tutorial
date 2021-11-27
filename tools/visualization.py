
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error



def plot_prediction(y_true, y_pred, title, fig_save_folder, fig_name):
    """
    plot predicted values vs true values

    Input Parameters:
    ----------------
    y_true : pandas series, numpy array
        true or experimental values 
    
    y_pred : pandas series, numpy array
        predicted values     
    
    title: string
    
    fig_save_folder, fig_name : string
        folder and figure name to be saved
    """
    
    R2 = r2_score(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))

    sns.set_style('darkgrid')

    min_ = int(np.min([np.min(y_true), np.min(y_pred)])) - 1
    max_ = int(np.max([np.max(y_true), np.max(y_pred)])) + 2
    x=[min_, max_]
    fig = plt.figure(figsize=(6, 6), facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(y_true, y_pred, 'o')
    ax.plot(x, x, linestyle='--', label=r'$y=x$')
    ax.set_xlabel('True Value', size=20)
    ax.set_ylabel('Predicted Value', size=20)
    #ax.set_title(f'{title}\n' + '$R^2$ = {:.3f}'.format(R2), size=20)
    ax.set_title('{}\n$R^2$ = {} , RMSE = {}'.format(title, round(R2, 3), round(RMSE,3)),fontsize=20)
    ax.set_xlim([min_, max_])
    ax.set_ylim([min_, max_])
    ax.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig(f'{fig_save_folder}/{fig_name}.png')
    #plt.show()   # must be after "plt.savefig"
    matplotlib.rc_file_defaults()
    #plt.close()