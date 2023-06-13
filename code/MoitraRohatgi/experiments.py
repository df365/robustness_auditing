import numpy as np
import matplotlib.pyplot as plt
import examples
import algorithms
import barplot

def compute_error_bars(x,low,high):
    n = len(x)
    error_bars = np.zeros((2,n))
    for i in range(n):
        error_bars[0][i] = np.median(x[i]) - np.percentile(x[i],low)
        error_bars[1][i] = np.percentile(x[i],high) - np.median(x[i])
    return error_bars

def plot_with_error_bars(x,y,low,high,name,col):
    error_bars = compute_error_bars(y,low,high)
    median_y = [np.median(l) for l in y]
    plt.errorbar(x,median_y,yerr=error_bars,label=name,color=col,capsize=3)

def heterogeneous_experiment(trials=10):
    klist = [10,64,118,173,227,282,336,391,445,500]
    gvals = []
    vals = []
    achieve_vals = []
    baseline_vals = []
    for k in klist:
        noise = 0.01
        gl=[]
        vl=[]
        al=[]
        bl=[]
        for j in range(trials):
            X,y=examples.heterogeneous_data(1000,k,noise)
            gl.append(algorithms.sensitivity(X,y))
            vl.append(algorithms.lp_algorithm_2d(X,y,[-noise,0,noise],30))
            al.append(algorithms.net_algorithm(X,y,1000))
            bl.append(algorithms.certify_by_residual_2d(X,y,[-noise,0,noise],1000))
        gvals.append(gl)
        vals.append(vl)
        achieve_vals.append(al)
        baseline_vals.append(bl)
    plot_with_error_bars(klist,gvals,25,75,"Greedy upper bound","grey")
    plot_with_error_bars(klist,achieve_vals,25,75,"Net upper bound","blue")
    plot_with_error_bars(klist,vals,25,75,"LP lower bound","red")
    plot_with_error_bars(klist,baseline_vals,25,75,"Baseline lower bound","black")
    plt.xlabel("Number of planted samples")
    plt.ylabel("Sample weight removed")
    plt.legend()
    plt.show()

def iso_experiment_2d(trials=10):
    noise_list = np.logspace(-1,1,10)
    gvals = []
    vals = []
    achieve_vals = []
    baseline_vals = []
    for i in range(len(noise_list)):
        noise = noise_list[i]
        gl = []
        vl = []
        al = []
        bl = []
        for j in range(trials):
            X,y=examples.isotropic_gaussian_data(1000,2,noise)
            gl.append(algorithms.sensitivity(X,y))
            vl.append(algorithms.lp_algorithm_2d(X,y,[-noise,0,noise],30))
            al.append(algorithms.net_algorithm(X,y,100))
            bl.append(algorithms.certify_by_residual_2d(X,y,[-noise,0,noise],1000))
        gvals.append(gl)
        vals.append(vl)
        achieve_vals.append(al)
        baseline_vals.append(bl)
    plot_with_error_bars(noise_list,gvals,25,75,"Greedy upper bound","grey")
    plot_with_error_bars(noise_list,achieve_vals,25,75,"Net upper bound","blue")
    plot_with_error_bars(noise_list,vals,25,75,"LP lower bound","red")
    plot_with_error_bars(noise_list,baseline_vals,25,75,"Baseline lower bound","black")
    plt.xlabel("Noise level")
    plt.ylabel("Sample weight removed")
    plt.xscale("log")
    plt.legend()
    plt.show()

def iso_experiment_3d():
    noise_list = np.logspace(-1,1,10)
    gvals = []
    vals = []
    achieve_vals = []
    for i in range(len(noise_list)):
        noise = noise_list[i]
        X,y=examples.isotropic_gaussian_data(500,3,noise)
        gvals.append(algorithms.sensitivity(X,y))
        vals.append(algorithms.lp_algorithm(X,y,[-noise,0,noise],30))
        achieve_vals.append(algorithms.net_algorithm(X,y,1000))
    plot_with_error_bars(noise_list,gvals,25,75,"Greedy upper bound","grey")
    plot_with_error_bars(noise_list,achieve_vals,25,75,"Net upper bound","blue")
    plot_with_error_bars(noise_list,vals,25,75,"LP lower bound","red")
    plt.xlabel("Noise level")
    plt.ylabel("Sample weight removed")
    plt.xscale("log")
    plt.legend()
    plt.show()


def covariance_shift_experiment():
    X, y = examples.covariance_shift_data(1000,30,0.2,300)
    print("Greedy upper bound:", algorithms.sensitivity(X,y))
    print("Net upper bound:", algorithms.net_algorithm(X,y,1000))

def boston_housing_split(random=False):
    X, y = examples.boston_housing_data(range(13))
    covnames = examples.boston_housing_features() + ['const']
    X0 = np.zeros((506,14))
    X0.T[13] = np.ones((506))
    for i in range(13):
        X0.T[i] = X.T[i] - np.mean(X.T[i])
        X0.T[i] = X0.T[i] / np.std(X0.T[i])
    wAll = np.ones((506))
    wRandom = np.random.binomial(1,134/506.,506)
    wSuburbs = np.zeros((506))
    for i in range(506):
        if X[i][1] > 5:
            wSuburbs[i] = 1
    wCity = 1 - wSuburbs
    assert(sum(wSuburbs) == 134)
    if random:
        barplot.bar_plot(plt,{"all": algorithms.ols(X0,y,wAll), "Ber(0.26)": algorithms.ols(X0,y,wRandom), "1-Ber(0.26)": algorithms.ols(X0,y,1-wRandom)})
    else:
        barplot.bar_plot(plt,{"all": algorithms.ols(X0,y,wAll), "suburbs": algorithms.ols(X0,y,wSuburbs), "city": algorithms.ols(X0,y,wCity)})
    plt.xticks(range(14), covnames)
    plt.show()

def boston_housing_experiment(*which):
    if 0 in which:
        # three-feature experiment
        X, y = examples.boston_housing_data([1,5,8])
        print("Greedy upper bound:", algorithms.sensitivity(X,y))
        print("Net upper bound:", algorithms.net_algorithm(X,y,1000))
        print("LP lower bound:", algorithms.lp_algorithm(X,y,[0],30))
    if 1 in which:
        # zn/crim experiment
        X, y = examples.boston_housing_data([1, 0])
        print("Greedy upper bound:", algorithms.sensitivity(X,y))
        print("Net upper bound:", algorithms.net_algorithm(X,y,1000))
        print("LP lower bound:", algorithms.lp_algorithm_2d(X,y,[0],100))
        plt.scatter(X.T[0],X.T[1],color='orange',marker='+')
        plt.xlabel("Fraction of residential land zoned for large lots")
        plt.ylabel("Per capita crime rate")
        plt.yscale("log")
        plt.show()
        boston_housing_split(random=False)
        boston_housing_split(random=True)
    if 2 in which:
        # all pairs experiment
        greedy_values = []
        net_values = []
        lp_values = []
        for a in range(13):
            for b in range(13):
                if a == b:
                    continue
                X,y = examples.boston_housing_data([a,b])
                greedy_values.append(algorithms.sensitivity(X,y))
                net_values.append(algorithms.net_algorithm(X,y,100))
                lp_values.append(algorithms.lp_algorithm_2d(X,y,[0],100))
        plt.scatter(greedy_values, net_values)
        plt.xlabel("# samples removed by greedy")
        plt.ylabel("# samples removed by net")
        plt.show()
        plt.scatter(net_values, lp_values)
        plt.xlabel("# samples removed by net")
        plt.ylabel("LP lower bound")
        plt.show()
