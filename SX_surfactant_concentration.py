import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy import stats

plt.rcParams['font.size'] = 8
rcParams['mathtext.default'] = 'regular'
pd.options.mode.chained_assignment = None  # default='warn'

# =============================================================================
#
# DATA ACQUISITION
#
# =============================================================================

# Data file: csv format with 14 columns
# First Column: Sample label 
#   starts with element name (Er or Nd) or Mixture (Er and Nd)
#   surfactant concentration, mixing time comes next but random order
#   surfactant concentration starts with "r="
#   Mixing time sometimes only contain number or ends with "min"
# 2~7th Column: Nd mass spectroscopy data
#   Isotope 142~148 (no 147) analyzed
#   concentration in ppb (float)
# 8~10th Column: Er mass spectroscopy data
#   Isotope 166~168 Analyzed
#   concentration in ppb (float)
# 11, 12th Column: Averaged Nd and Er concentration in ppb
#   These columns are not needed since Dilution factor should be considered first & I am interested in M unit
# 13, 14th Column: Dilution
#   Extracted sample is diluted with 3% HNO3
#   extracted: extracted sample mass in g
#   HNO3: mass of 3% HNO3 added for dilution in g

# =============================================================================
# 
# GOAL
# 
# =============================================================================

# Data Cleaning
# Decouple components in the "Sample" column:
#   Sample type ("Pure", "Mixture"): pure solution contains only one element, whereas mixture has both
#   element ("Er","Nd")
#   surfactant concentration: "r=#" (as a string to make plotting stage easier)
#   Mixing time: in integer or float64

# Data Analysis
# The numbers in data: remaining (unextracted) ion concentration -> switch it to "Extracted" concentration
# extracted concentration = 1-unextracted/initial (x100%)
# Get 1. extracted concentration, 2. extracted percentage
# Check the reaction order: Plot log(c/c0) and c0/c with linear regression, Manually check the linearity
#   If linear in log plot: 1st order reaction c = c0*exp(-kt)
#   If linear in 1/c plot: 2nd order reaction dc/dt ~ c^2
# 
# Note!!!
# In each data file (csv file) there are stock solution entries
# The concentrations in a csv file should be analyzied with the stock solution concenration in the "SAME FILE"


def get_data_conc():     # Return or generate global variables of Extraction result in concentration
    path = 'extraction_results/'
    files = os.listdir(path)    # List of csv file names in the path
    df_final = pd.DataFrame(columns=["sample_type",'r',"mixing time",'Er_conc','Nd_conc'])
    for file in files:
        if file.endswith('.csv') == True:   # Read csv files only
            df=pd.read_csv(path+file,skiprows=1,index_col=False)
                        
            # Data Cleaning
            df.columns = df.columns.str.replace(" ","")             # Data cleaning: remove space after element
            df["Sample"] = df["Sample"].str.replace(" ","_")        # Data cleaning: replace space with _ in sample label
            df["Sample"] = df["Sample"].str.replace("_min","min")   # Data cleaning: 3 min -> 3_min -> 3min
            
            # Make sample type, element, r columns based on "Sample" values
            # Sample label starts with Er, Nd, Mix, or Mixture
            # Sample Type: assign Mixture or Pure (if Er or Nd)
            # Assign element columns with boolean values: if True, that element exists in the sample
            def label_sample(row):
                if row["Sample"].split('_')[0] == 'Mix' or row["Sample"].split('_')[0] == 'Mixture':
                    return ("Mixture", True, True)
                elif row["Sample"].split('_')[0] == 'Er':
                    return ("Pure", True, False)
                elif row["Sample"].split('_')[0] == 'Nd':
                    return ("Pure", False, True)
            # Make sample_type column (pure or mixture)
            df["sample_type"]=df.apply(lambda row:label_sample(row)[0],axis=1)
            df["Er"]=df.apply(lambda row:label_sample(row)[1],axis=1)
            df["Nd"]=df.apply(lambda row:label_sample(row)[2],axis=1)
            
            # Sampe col contains information of r and t
            # r always starts with 'r='
            def label_r_and_t(row):
                if 'r=' in row["Sample"]:
                    for strs in row["Sample"].split('_')[1:]:
                        if strs.startswith('r=') == True:
                            r_value = strs
                        else: # if not starts with 'r=': information about t!
                            t_value = strs
                            if t_value.startswith('t='):
                                t_value = t_value.replace('t=','')
                                #t_value = int(t_value)
                            if t_value.endswith('min'):
                                t_value = t_value.replace('min','')
                                #t_value = int(t_value)
                    return (r_value, int(t_value))
                else:
                    return ('stock',0)
                
            df["r"]=df.apply(lambda row: label_r_and_t(row)[0], axis=1)
            df["mixing time"] = df.apply(lambda row: label_r_and_t(row)[1], axis=1)
            
            # Get extracted percentage for each element            
            df["dilute"]=(df['extracted']+df['HNO3'])/df['extracted']
            def get_Er_conc(row):
                if row["Er"]==True:
                    return row['dilute']*(row['166Er']/166+row['167Er']/167+row['168Er']/168)/3
            def get_Nd_conc(row):
                if row['Nd']==True:
                    return row['dilute']*(row['142Nd']/142+row['143Nd']/143+row['145Nd']/145+row['146Nd']/146+row['148Nd']/148)/6
            
            df['Er_conc'] = df.apply(lambda row:get_Er_conc(row),axis=1)
            df['Nd_conc'] = df.apply(lambda row:get_Nd_conc(row),axis=1)            
            df = df[["sample_type",'r',"mixing time",'Er_conc','Nd_conc']]
            
            list_r_df = [] # List of unique r values without 'stock' in "Single" DataFrame.
            for r_val in df['r'].unique():
                if r_val == 'stock':
                    pass
                else:
                    list_r_df.append(r_val)
            
            temp = df[df['r']=='stock']
            
            if len(list_r_df)==1:
                temp1 = temp
                for j in range(len(temp1)):
                    temp1['r'].iloc[j] = list_r_df[0]
                    
            for i in range(len(list_r_df)):
                temp_i = temp.replace({'stock':list_r_df[i]})
                df = pd.concat([df,temp_i],axis=0)
            
            df_final = pd.concat([df_final,df],axis=0,ignore_index=True)
                            
        # Dealing with Stock Solution Samples
        # The concentration of stock solution -> assign to r=# t=0
    df_final = df_final.drop(df_final[df_final['r']=='stock'].index)
    df_final.sort_values(by=['r','mixing time'],inplace=True,axis=0,ignore_index=True)
    
    return df_final

def call_data(df, sam_type, element, r):
    # How will I call the data?
    # Passing string of type element r
    # Return the extracted percentage
    temp = pd.DataFrame(columns=['time','conc','perc'])
    time = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())]['mixing time']
    conc = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())][f'{element}_conc']
    time = time.astype('float')
    temp['time']=time
    temp['conc']=conc
    
    stock_conc=temp[temp['time']==0]['conc']
    
    def conc_to_per(row):
        return 100 * (stock_conc - row['conc'])/stock_conc
    
    temp['perc'] = temp.apply(lambda row: conc_to_per(row), axis=1)
    #print(time.values, temp['perc'].values)
    
    return time.values, temp['perc'].values

def call_conc(df, sam_type, element, r):
    # How will I call the data?
    # Passing string of type element r
    # Return the extracted percentage
    time = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())]['mixing time']
    conc = df[(df['sample_type']==sam_type)&(df['r']==r)&(df[f'{element}_conc'].notnull())][f'{element}_conc']
    time = time.astype('float')
    return time.values, conc.values


# =============================================================================
# 
# DATA ANALYSIS PART
# 
# =============================================================================

def curve_peleg(x,k1,k2):   # Langmuir isotherm adsorption
    return x/(1/k1+x/k2)

def curve_exp(x,c,k):   # 1st order reaction
    return c*(1-np.exp(-x*k))

def curve_exp_modified(x,c,k,b):    # 1st order reaction with base extraction
    return c*(1-np.exp(b-x*k))

def curve_2exp(x,c_1,k_1,c_2,k_2):  # 1st order reaction with two mechanism
    return c_1*(1-np.exp(-x*k_1)) + c_2*(1-np.exp(-x*k_2))


def fit_exp(t,data):
    x=np.linspace(0,45,100)
    data_fit=data[1:]
    t_fit=t[1:]
    err=2**0.5 * (1-0.01*np.array(data_fit))
    popt, pcov= curve_fit(curve_exp,t_fit,data_fit,sigma=err,maxfev=10000,
                          bounds=([0,0],[100,np.inf]))
    y=curve_exp(x, *popt)
    return x,y, popt,pcov

def fit_exp_minmax(t,data,k_range):
    x=np.linspace(0,45,100)
    data_fit=data[1:]
    t_fit=t[1:]
    err=2**0.5 * (1-0.01*np.array(data_fit))
    popt, pcov= curve_fit(curve_exp,t_fit,data_fit,sigma=err,maxfev=10000,
                          bounds=([0,k_range[0]],[100,k_range[1]]))
    y=curve_exp(x, *popt)
    return x,y, popt,pcov

def fit_exp_modified(t,data,k_range,b_range):
    x=np.linspace(0,45,100)
    data_fit=data[1:]
    t_fit=t[1:]
    err=2**0.5 * (1-0.01*np.array(data_fit))
    popt, pcov= curve_fit(curve_exp_modified,t_fit,data_fit,sigma=err,maxfev=10000,
                          bounds=([0,k_range[0],min(b_range)],[100,k_range[1],max(b_range)]))
    y=curve_exp_modified(x, *popt)
    return x,y, popt,pcov  

def fit_2exp(t,data):
    x=np.linspace(0,45,100)
    data_fit=data[1:]
    t_fit=t[1:]
    err=2**0.5 * (1-0.01*np.asarray(data_fit))
    popt, pcov = curve_fit(curve_2exp,t_fit,data_fit,sigma=err,maxfev=10000,
                           bounds=([0,0,0,0],[max(data[-1],10),np.inf,max(data[-1],10),np.inf]))
    y=curve_2exp(x, *popt)
    return x,y, popt, pcov

def get_chisq(t, data, model, popt):    # In the case you want to automate the model selection
    t_fit=t[1:]
    data_fit=data[1:]
    t_fit=np.asarray(t_fit)
    data_fit = np.asarray(data_fit)
    err=2**0.5*(1-0.01*np.asarray(data_fit))
    
    if str(model)=='1exp':
        y=curve_exp(t_fit,*popt)
    elif str(model)=='2exp':
        y=curve_2exp(t_fit,*popt)
    elif str(model)=='1exp_modified':
        y=curve_exp_modified(t_fit,*popt)
    #print(data_fit.shape, y.shape, err.shape)
    chisq = (data_fit-y)**2/err**2
    return sum(chisq)


def model_selection(t,data):
    for model in model_cand:
        if model == '1exp':
            x,y,popt,pcov=fit_exp(t,data)
            chisq_1exp = get_chisq(t,data, model, popt)
        elif model == '2exp':
            x,y,popt,pcov=fit_2exp(t,data)
            chisq_2exp = get_chisq(t,data, model, popt)
        else:
            print('Update: model_cand')
    
    if chisq_1exp/chisq_2exp > 5:
        return '2exp'
    else:
        return '1exp'

def log_plot(df):
    r_cand = ['r=3','r=4.5','r=6']
    k=[]
    b=[]
    r_sq=[]
    sam=[]
    for sam_type in ['Pure']:#df['sample_type'].unique():
        for element in ['Er','Nd']:
            for r in r_cand:
                fig=plt.figure()
                fig_width=3.25
                fig_height=3.25/1.618
                fig.set_size_inches(fig_width,fig_height)
                # Plot Experiment data
                t, c = call_conc(df,sam_type, element, r)
                ax=fig.add_axes([0.2,0.2,0.7,0.7], xlim = (-1.5,45))
                ax.plot(t,np.log((c/c[0]-(1-0.89534)).astype('float64')),linewidth=0,marker='o',markersize=3)
                
                if sam_type == 'Pure' and element == 'Er' and r == 'r=3':
                    # Fetch first 4 points
                    t_temp=t[4:]
                    c_temp=c[4:]
                    slope, intercept, rvalue, pvalue, std_err = stats.linregress(t_temp, np.log((c_temp/c[0]).astype('float64')))
                    ax.text(0,-6,f'y=k*x+b\nk={slope:.5f}\nb={intercept:.5f}\nR$^2$={rvalue**2:.5f}',va='bottom',ha='left',color='b')
                    x=np.linspace(10,45)
                    y=slope*x+intercept
                    ax.plot(x,y,linewidth=1,linestyle='--',color='b')
                    #ax.axhline(np.log(1-0.89534),0,1, linewidth=0.5, linestyle='--', color='k')
                    
                    t_temp=t[:4]
                    c_temp=c[:4]
                    slope, intercept, rvalue, pvalue, std_err = stats.linregress(t_temp, np.log((c_temp/c[0]).astype('float64')))
                else:
                    t_temp=t[:]
                    c_temp=c[:]
                    slope, intercept, rvalue, pvalue, std_err = stats.linregress(t_temp, np.log((c_temp/c[0]).astype('float64')))
                
                # Plot fit
                x=np.linspace(0,45)
                y=slope*x+intercept
                ax.plot(x,y,color='k',linewidth=1,linestyle='--')
                ax.text(44,0,f'y=k*x+b\nk={slope:.5f}\nb={intercept:.5f}\nR$^2$={rvalue**2:.5f}',va='top',ha='right')
                
                ax.set_title(f'Log {sam_type}_{element}_{r}')
                ax.set_ylabel('log(c/c0)')
                ax.set_xlabel('mixing time(min)')
                plt.savefig(f'log_plot/{sam_type}_{element}_{r}.png', dpi=1000)
                
                # Save fit result
                k.append(slope.astype('float64'))
                b.append(intercept.astype('float64'))
                r_sq.append((rvalue**2).astype('float64'))
                sam.append(f'{sam_type} {element} {r}')
    d = {'sample':sam, 'k':k, 'b':b, 'r_sq':r_sq}
    df_para = pd.DataFrame(data=d)
    df_para['exp(b)']=np.exp(df_para['b'])
    df_para.to_csv('linfit_para.csv')

def rec_plot(df):
    r_cand = ['r=3','r=4.5','r=6']
    for sam_type in ['Pure']:#df['sample_type'].unique():
        for element in ['Er','Nd']:
            for r in r_cand:
                fig=plt.figure()
                fig_width=3.25
                fig_height=3.25/1.618
                fig.set_size_inches(fig_width,fig_height)
                t, c = call_conc(df,sam_type, element, r)
                
                ax=fig.add_axes([0.2,0.2,0.7,0.7], xlim = (-1.5,45))
                ax.plot(t,c[0]/c,linewidth=0,marker='o')
                ax.set_title(f'1/Ln {sam_type}_{element}_{r}')
                plt.savefig(f'reciprocal_plot/{sam_type}_{element}_{r}.png', dpi=1000)

def analysis_plot(df):
    r_cand = ['r=3','r=4.5','r=6']
    #d={}
    for r in r_cand:
        for sam_type  in ['Pure']:#df['sample_type'].unique():
            for element in ['Er', 'Nd']:
                fig=plt.figure()
                fig_width=3.25
                fig_height=3.0*3.25/1.618
                fig.set_size_inches(fig_width,fig_height)
                
                # Set Axis and Title
                ax1=fig.add_axes([0.2,0.2/3,0.7,0.7/3], xlim=(-1.5,45), ylim=(-5,105))     # two exp
                ax2=fig.add_axes([0.2,1.1/3,0.7,0.7/3], xlim=(-1.5,45), ylim=(-5,105))     # one exp
                ax3=fig.add_axes([0.2,2.1/3,0.7,0.7/3], xlim=(-1.5,45), ylim=(-5,105))
                ax3.set_title(f'{sam_type} {element} {r}\n 1exp')
                ax2.set_title('Modified 1exp: y=E*(1-e$^b$ e$^{-kx}$)')
                ax1.set_title('2exp')
                
                # Call and plot Data
                t, data = call_data(df, sam_type, element, r)
                ax1.plot(t,data,linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
                ax2.plot(t,data,linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
                ax3.plot(t,data,linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
                
                # Fit 2exp and Plot the fit
                x,y,popt,pcov=fit_2exp(t,data)
                c1, k1, c2, k2 = popt[0], popt[1], popt[2], popt[3]
                perr=np.sqrt(np.diag(pcov))
                sig_c1, sig_k1, sig_c2, sig_k2 = perr[0], perr[1], perr[2], perr[3]
                
                ax1.plot(x,y,linewidth=1)
                ax1.plot(x,c1*(1-np.exp(-x*k1)), linestyle='--', linewidth=0.5, color='tab:green')
                ax1.plot(x,c2*(1-np.exp(-x*k2)), linestyle='--', linewidth=0.5, color='tab:red')
                
                X2 = get_chisq(t,data,'2exp',popt)
                ax1.axhline(popt[0]+popt[2],0,1, linewidth=0.5, linestyle='--', color='k')
                ax1.text(44,0,
                         #f'c1:{c1:.3f}+/-{sig_c1:.3f}  k1:{k1:.3f}+/-{sig_k1:.3f}\nc2:{c2:.3f}+/-{sig_c2:.3f}  k2:{k2:.3f}+/-{sig_k2:.3f}',
                         f'c1:{c1:.3f}  k1:{k1:.3f}\nc2:{c2:.3f}  k2:{k2:.3f}',
                         va='bottom',ha='right')
                ax1.text(0,100,
                         #'E:{}+/-{} \n $\\chi^2$={}'.format(round(popt[0]+popt[2],2),round(np.sqrt(pcov[0,0]+pcov[2,2]),2),round(X2,2)),
                         'E:{}\n $\\chi^2$={}'.format(round(popt[0]+popt[2],2),round(X2,2)),
                         va='top',ha='left')
                
                df_linfit = pd.read_csv('linfit_para.csv')
                # Fit 1exp and Plot
                if f'{sam_type} {element} {r}' in df_linfit['sample'].tolist():
                    k_val = df_linfit[df_linfit['sample']==f'{sam_type} {element} {r}']['k'].values[0]
                    k_range=[0.999*abs(k_val),1*abs(k_val)]
                    #print(f'{sam_type} {element} {r}'+str(k_range))
                    x,y,popt,pcov = fit_exp(t,data)#,k_range)
                
                else:
                    x,y,popt,pcov = fit_exp(t,data)
                
                c, k = popt[0], popt[1]
                perr=np.sqrt(np.diag(pcov))
                sig_c, sig_k = perr[0], perr[1]
                X2=get_chisq(t,data,'1exp',popt)
                
                ax3.plot(x,y,linewidth=1,color='k')
                ax3.axhline(popt[0],0,1, linewidth=0.5, linestyle='--', color='k')
                ax3.text(44,1,
                         #f'E: {c:.3f} +/- {sig_c:.3f}\nk: {k:.3f} +/- {sig_k:.3f}\n $\\chi^2$={X2:.3f}',
                         f'E: {c:.3f}\nk: {k:.3f}\n $\\chi^2$={X2:.3f}',
                         ha='right',va='bottom')
                
                # Fit modified 1exp and plot
                k_val = df_linfit[df_linfit['sample']==f'{sam_type} {element} {r}']['k'].values[0]
                b_val = df_linfit[df_linfit['sample']==f'{sam_type} {element} {r}']['b'].values[0]
                k_range=[0.999*abs(k_val),1.0*abs(k_val)]
                b_range=[0.999*b_val, 1.0*b_val]
                x,y,popt,pcov = fit_exp_modified(t,data,k_range,b_range)              
                c,k,b = popt[0], popt[1], popt[2]
                perr=np.sqrt(np.diag(pcov))
                sig_c, sig_k, sig_b = perr[0], perr[1], perr[2]
                X2=get_chisq(t,data,'1exp_modified',popt)
                ax2.plot(x,y,linewidth=1,color='k')
                ax2.axhline(popt[0],0,1, linewidth=0.5, linestyle='--', color='k')
                ax2.text(44,1,
                         #f'E: {c:.3f} +/- {sig_c:.3f}\nk: {k:.3f} +/- {sig_k:.3f}\n $\\chi^2$={X2:.3f}',
                         f'E: {c:.3f}\nk: {k:.3f}\nb: {b:.3f}\n$\\chi^2$={X2:.3f}',
                         ha='right',va='bottom')
                
                plt.savefig(f'model_selection/{sam_type}_{element}_{r}.png',dpi=600)

# =============================================================================
# 
# PUBLICATION FIGURES
# 
# =============================================================================

def plot_pure_log(df):
    fig = plt.figure()
    fig_width=3.25
    fig_height=3.25
    fig.set_size_inches(fig_width,fig_height)
    
    ax=fig.add_axes([0.2,0.2,0.7,0.7], xlim=(-1.5,45))
    ax.set_xlabel('Mixing Time (min)')
    ax.set_ylabel('log (c/c$_{0}$)')
    #ax.text(0,-6,'Fit with\ny=-kx+b')
    r_cand = ['r=3', 'r=4.5', 'r=6']

    for sam_type in ['Pure']:#df['sample_type'].unique():
        for element in ['Er','Nd']:
            for r in r_cand:
                t, c = call_conc(df,sam_type, element, r)
                if element == 'Er':
                    color='r'
                elif element == 'Nd':
                    color='g'
                    
                if r == 'r=3':
                    marker='^'
                elif r == 'r=4.5':
                    marker='o'
                elif r== 'r=6':
                    marker='s'
                ax.plot(t,np.log((c/c[0]).astype('float64')),linewidth=0,
                        marker=marker,color=color,markersize=3,
                        label=f'{element} {r}')
                if sam_type == 'Pure' and element == 'Er' and r == 'r=3':
                    # Fetch first 4 points
                    t_temp=t[:4]
                    c_temp=c[:4]
                    slope, intercept, rvalue, pvalue, std_err = stats.linregress(t_temp, np.log((c_temp/c[0]).astype('float64')))
                else:
                    t_temp=t[:]
                    c_temp=c[:]
                    slope, intercept, rvalue, pvalue, std_err = stats.linregress(t_temp, np.log((c_temp/c[0]).astype('float64')))
                
                # Plot fit
                x=np.linspace(-1,45)
                y=slope*x+intercept
                ax.plot(x,y,color='k',linewidth=1,linestyle='--',zorder=0,alpha=0.8)
        
    ax.text(44,-0.2,'Fitted with y=-kt+b',ha='right',va='bottom',weight='bold')
    ax.axhline(np.log(1-0.89534),0,1, linewidth=1, linestyle=':', color='k')
    ax.text(44,np.log(1-0.91),'Equilibrium\n of Er r=3',ha='right',va='top')
    ax.legend()        
    plt.savefig('logplot.png',dpi=1000)

def plot_pure(df):
    c = ['tab:blue','tab:orange','tab:green','tab:brown','tab:red']
    if len(c) != len(df['r'].unique()):
        print('Not enough colors')
    fig=plt.figure()
    fig_width = 3.25
    fig_height= 2.0 * fig_width / (1.618)
    fig.set_size_inches(fig_width,fig_height)
    ### Imaginary axis for axis-label
    ax3 = fig.add_axes([0.14,0.2,0.76,0.75])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.set_ylabel('Extracted Percentage (%)')
    ### Bottom    
    ax1 = fig.add_axes([0.2,0.2,0.7,0.35], xlim=(-1.5,45), ylim=(-5,105))
    ax1.tick_params(direction='in', length=3)
    ax1.set_xlabel('Mixing Time (min)')
    ### Top
    ax2 = fig.add_axes([0.2,0.6,0.7,0.35], xlim=(-1.5,45), ylim=(-5,105))
    ax2.tick_params(direction='in', length=3)
    
    col_idx = 0
    df_linfit = pd.read_csv('linfit_para.csv')
    sam_type='Pure'
    r_cand = ['r=0','r=1','r=3','r=4.5','r=6'] #'r=0','r=1',
    for r in r_cand:
        # Plot Er
        t, data = call_data(df, sam_type, 'Er', r)
        ax2.errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color=c[col_idx],
                 linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
        
        if r == 'r=3' or r=='r=4.5' or r=='r=6':
            k_val = df_linfit[df_linfit['sample']==f'{sam_type} Er {r}']['k'].values[0]
            b_val = df_linfit[df_linfit['sample']==f'{sam_type} Er {r}']['b'].values[0]
            k_range=[0.999*abs(k_val),1.0*abs(k_val)]
            b_range=[0.999*b_val, 1.0*b_val]
            x,y, popt,pcov = fit_exp_modified(t, data, k_range, b_range)
            ax2.plot(x,y,color=c[col_idx],linewidth=1)
        
        
        model = model_selection(t,data)
        if model == '1exp':
            x,y,popt,pcov = fit_exp(t,data)
            ax2.plot(x,y,linewidth=1, color=c[col_idx])
        elif model == '2exp':
            x,y,popt,pcov = fit_2exp(t,data)
            ax2.plot(x,y,linewidth=1, color=c[col_idx])
        print(f'{sam_type} Er {r} model: {model}')
        
        #Plot Nd
        t, data = call_data(df, sam_type, 'Nd', r)
        ax1.errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color=c[col_idx],
                 linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
        model = model_selection(t,data)
        if model == '1exp':
            x,y,popt,pcov = fit_exp(t,data)
            ax1.plot(x,y,linewidth=1, color=c[col_idx])
        elif model == '2exp':
            x,y,popt,pcov = fit_2exp(t,data)
            ax1.plot(x,y,linewidth=1, color=c[col_idx])
        print(f'{sam_type} Nd {r} model: {model}')
        
        col_idx+=1
    

    legends = []
    for r in r_cand:
        legends.append(f'{r}')
    ax1.legend(legends,loc='lower center', bbox_to_anchor=(0.5, -0.6),ncol=3,handletextpad=0.1)
    ax1.text(-0.5,102.5,'(d)', weight='bold', va='top', ha='left')
    ax1.text(3.5,103,'Nd only', weight='bold', va='top', ha='left',fontsize=10)
    ax2.text(-0.5,102.5,'(c)', weight='bold', va='top', ha='left')
    ax2.text(3.5,103,'Er only', weight='bold', va='top', ha='left',fontsize=10)
    plt.savefig('extraction_plot/Pure_cases11.png', dpi=1000)

def plot_pure_log_included(df):
    fig = plt.figure()
    fig_width = 7
    fig_height = fig_width / (1.618)
    fig.set_size_inches(fig_width, fig_height)
    
    # Axis properties
    ax1 = fig.add_axes([0.1,0.55,0.35,0.35],xlim=(-1.5,45), ylim=(-5,105))
    ax1.tick_params(direction='in', length=3)
    ax1.set_ylabel('Extraction Percentage (%)')
    #ax1.set_xlabel('Mixing Time (min)')
    ax1.text(-0.5,102.5,'(a)', weight='bold', va='top', ha='left')
    ax1.text(3.5,102.5,'Er only', weight='bold', va='top', ha='left',fontsize=10)
    
    ax2 = fig.add_axes([0.1,0.15,0.35,0.35],xlim=(-1.5,45), ylim=(-5,105))
    ax2.tick_params(direction='in', length=3)
    ax2.set_ylabel('Extraction Percentage (%)')
    ax2.set_xlabel('Mixing Time (min)')
    ax2.text(-0.5,102.5,'(b)', weight='bold', va='top', ha='left')
    ax2.text(3.5,102.5,'Nd only', weight='bold', va='top', ha='left',fontsize=10)
    
    ax3 = fig.add_axes([0.5,0.7,0.4,0.2],xlim=(0,1),ylim=(0,1))
    ax3.text(-0.01,0.95,'(c)',weight='bold',va='top', ha='right')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    ax4 = fig.add_axes([0.55,0.1,0.35,0.55])
    ax4.text(-10,0.3,'(d)', weight='bold',va='top',ha='right')
    ax4.set_xlabel('Mixing Time (min)')
    ax4.set_ylabel('log (c/c$_{0}$)')
    
    # ax1: Er extraction
    # ax2: Nd extraction
    # ax3: Cartoon expression interfacial structure
    # ax4: log
    
    def plot_pure_extraction(ax1,ax2):
        c = ['tab:blue','tab:orange','tab:green','tab:brown','tab:red']
        col_idx = 0
        df_linfit = pd.read_csv('linfit_para.csv')
        sam_type='Pure'
        r_cand = ['r=0','r=1','r=3','r=4.5','r=6'] #'r=0','r=1',
        for r in r_cand:
            # Plot Er
            t, data = call_data(df, sam_type, 'Er', r)
            ax1.errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color=c[col_idx],
                     linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
            if r=='r=4.5' or r=='r=6':
                k_val = df_linfit[df_linfit['sample']==f'{sam_type} Er {r}']['k'].values[0]
                b_val = df_linfit[df_linfit['sample']==f'{sam_type} Er {r}']['b'].values[0]
                k_range=[0.999*abs(k_val),1.0*abs(k_val)]
                b_range=[0.999*b_val, 1.0*b_val]
                x,y, popt,pcov = fit_exp_modified(t, data, k_range, b_range)
                ax1.plot(x,y,color=c[col_idx],linewidth=1)
            else:
                x,y,popt,pcov = fit_exp(t,data)
                ax1.plot(x,y,linewidth=1, color=c[col_idx])
            
            #Plot Nd
            t, data = call_data(df, sam_type, 'Nd', r)
            ax2.errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color=c[col_idx],
                     linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1)
            x,y,popt,pcov = fit_exp(t,data)
            ax2.plot(x,y,linewidth=1, color=c[col_idx])
            
            col_idx+=1
        

        legends = []
        for r in r_cand:
            legends.append(f'{r}')
        ax2.legend(legends,loc='lower center', bbox_to_anchor=(0.5, -0.45),ncol=5,handletextpad=0.1,handlelength=1)
    
    def plot_log(ax):
        r_cand = ['r=3', 'r=4.5', 'r=6']
        for sam_type in ['Pure']:#df['sample_type'].unique():
            for element in ['Er','Nd']:
                for r in r_cand:
                    t, c = call_conc(df,sam_type, element, r)
                    if element == 'Er':
                        color='r'
                    elif element == 'Nd':
                        color='g'
                        
                    if r == 'r=3':
                        marker='^'
                    elif r == 'r=4.5':
                        marker='o'
                    elif r== 'r=6':
                        marker='s'
                    ax.plot(t,np.log((c/c[0]).astype('float64')),linewidth=0,
                            marker=marker,color=color,markersize=3,
                            label=f'{element} {r}')
                    if sam_type == 'Pure' and element == 'Er' and r == 'r=3':
                        # Fetch first 4 points
                        t_temp=t[:4]
                        c_temp=c[:4]
                        slope, intercept, rvalue, pvalue, std_err = stats.linregress(t_temp, np.log((c_temp/c[0]).astype('float64')))
                    else:
                        t_temp=t[:]
                        c_temp=c[:]
                        slope, intercept, rvalue, pvalue, std_err = stats.linregress(t_temp, np.log((c_temp/c[0]).astype('float64')))
                    
                    # Plot fit
                    x=np.linspace(-1,45)
                    y=slope*x+intercept
                    ax.plot(x,y,color='k',linewidth=1,linestyle='--',zorder=0,alpha=0.8)
            
        ax.text(44,-0.2,'Fitted with y=-kt+b',ha='right',va='bottom',weight='bold')
        ax.axhline(np.log(1-0.89534),0,1, linewidth=1, linestyle=':', color='k')
        ax.text(44,np.log(1-0.91),'Equilibrium\n of Er r=3',ha='right',va='top')
        ax.legend()        

    
    
    
    
    plot_pure_extraction(ax1,ax2)
    plot_log(ax4)
    plt.savefig('extraction_plot/pure_wide.png', dpi=1000)


def plot_mixture(df):
    ### Plot Axis ###
    fig = plt.figure()
    fig_width = 3.25
    aspect_ratio = 2.6
    fig_height = aspect_ratio * fig_width / (1.618)
    fig.set_size_inches(fig_width, fig_height)
    
    ### Bottom
    ax1 = fig.add_axes([0.2,0.2/aspect_ratio,0.7,0.7/aspect_ratio], xlim=(-1.5,45), ylim=(-5,105))
    ax1.tick_params(direction='in', length=3)
    ax1.set_xlabel('Mixing Time (min)')
    ax1.text(0,100,'r=6', weight='bold', va='top', ha='left', fontsize=10)
    ### Middle
    ax2 = fig.add_axes([0.2,1.0/aspect_ratio,0.7,0.7/aspect_ratio], xlim=(-1.5,45), ylim=(-5,105))
    ax2.tick_params(direction='in', length=3)
    ax2.set_ylabel('Extracted Percentage (%)')
    ax2.text(0,100,'r=3', weight='bold', va='top', ha='left', fontsize=10)
    ### Top
    ax3 = fig.add_axes([0.2,1.8/aspect_ratio,0.7,0.7/aspect_ratio], xlim=(-1.5,45), ylim=(-5,105))
    ax3.tick_params(direction='in', length=3)
    ax3.text(0,100,'r=1', weight='bold', va='top', ha='left', fontsize=10)
    
    # Plot only r=1, 3, and 6
    sam_type='Mixture'
    r_cand = ['r=1','r=3','r=6']
    axes = [ax3, ax2, ax1]
    
    for idx, r in enumerate(r_cand):
        # Plot Er
        element='Er'
        t,data = call_data(df, sam_type, element, r)
        axes[idx].errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color='r',
                           linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1,label='Er')
        model = model_selection(t,data)
        if model == '1exp':
            x,y,popt,pcov = fit_exp(t,data)
            axes[idx].plot(x,y,linewidth=1, color='r')
        elif model == '2exp':
            x,y,popt,pcov = fit_2exp(t,data)
            axes[idx].plot(x,y,linewidth=1, color='r')
        
        #Plot Nd
        element='Nd'
        t, data = call_data(df, sam_type, element, r)
        axes[idx].errorbar(t,data,yerr=2**0.5*(1-0.01*data),capsize=1,elinewidth=0.75, color='g',
                           linewidth=0,marker='o',markersize=5,markerfacecolor='None',markeredgewidth=1, label='Nd')
        model = model_selection(t,data)
        if model == '1exp':
            x,y,popt,pcov = fit_exp(t,data)
            axes[idx].plot(x,y,linewidth=1, color='g')
        elif model == '2exp':
            x,y,popt,pcov = fit_2exp(t,data)
            axes[idx].plot(x,y,linewidth=1, color='g')
    
    ax1.legend()
    
    
    plt.savefig('extraction_plot/Mixture_cases.png', dpi=1000)

df=get_data_conc()
#print(df)
rec_plot(df)
#log_plot(df)
#plot_pure(df)
#plot_mixture(df)
#analysis_plot(df)
#plot_pure_log(df)
#plot_pure_log_included(df)