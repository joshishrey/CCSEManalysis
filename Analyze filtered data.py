from os import listdir
from os.path import isfile, join
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numpy import array, sign, zeros
import numpy as np
import time
import seaborn as sns

mode='Atomizer'
font = {
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
file="Coated_cab_o_jet_atomizer_EDX1_Reduced.dat"
smps_data_path = 'AtmSMPS.csv'

headers = ["Id", "Area (μm²)", "Aspect Ratio", "Breadth (μm)", "Direction (°)", "ECD (μm)", 
           "Length (μm)", "Mean Gray Level", "Perimeter (μm)", "Shape", "Stage X", "Stage Y", 
           "Count", "C (Wt%)", "N (Wt%)", "O (Wt%)", "Na (Wt%)", "Mg (Wt%)", "Al (Wt%)", "Si (Wt%)", 
           "P (Wt%)", "S (Wt%)", "Cl (Wt%)", "K (Wt%)", "Ca (Wt%)", "Mn (Wt%)", "Fe (Wt%)", "Zn (Wt%)", 
           "Cl/Na", "O/Na", "C/Na", "O/C", "S/C", "O/S", "N/S", "Cl/(Na+0.5Mg)", "(Cl+N+0.5S)/(Na+0.5Mg)", "Class"]


# Reading the .dat file with the specified headers
df = pd.read_csv(file, sep='\t', header=None, names=headers)


try:
    df=df[pd.to_numeric(df['Id'], errors='coerce').notnull()]
except:
    print('cant find the Id column')

#Correct if the micron sign has differnt unicode
try:
    df['ECD (μm)']=df['ECD (µm)']
    df['Perimeter (μm)']=df['Perimeter (µm)']
    df['Area (μm²)']=df['Area (sq. µm)']
    df['Breadth (μm)']=df['Breadth (µm)']
    df['Length (μm)']=df['Length (µm)']
except:
    print('No chaning the microns required')

df['Sphericity']=np.pi*df['ECD (μm)']/df['Perimeter (μm)']


fig, ax = plt.subplots(1,1,figsize=(10,10))
sns.histplot(data=df, x="ECD (μm)",ax=ax,binwidth=0.01,binrange=[0.05,0.6] )
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
fig.suptitle(file, fontsize=16)
fig.savefig('ECD histogram,'+file+'.jpg')

EDX = df[['C (Wt%)', 'N (Wt%)', 'O (Wt%)', 'Na (Wt%)', 'Mg (Wt%)',
       'Al (Wt%)', 'Si (Wt%)', 'P (Wt%)', 'S (Wt%)', 'Cl (Wt%)', 'K (Wt%)',
       'Ca (Wt%)', 'Mn (Wt%)', 'Fe (Wt%)', 'Zn (Wt%)']]

fig, axEDX = plt.subplots(1,1,figsize=(10,10))
sns.boxplot(data=EDX,ax=axEDX)
axEDX.set_xticklabels(axEDX.get_xticklabels(),rotation=30)
axEDX.set(ylim=(0, 100))
fig.suptitle(file, fontsize=16)
fig.savefig('EDX %,'+file+'.jpg')

fig, axEDX = plt.subplots(1,1,figsize=(10,10))
sns.boxplot(data=EDX,ax=axEDX)
axEDX.set_xticklabels(axEDX.get_xticklabels(),rotation=30)
fig.suptitle(file, fontsize=16)
axEDX.set(ylim=(0, 20))
fig.savefig('EDX % 20,'+file+'.jpg')

#lets isolate physical features only for histogram
Phy = df[['Area (μm²)', 'Aspect Ratio', 'Breadth (μm)',  'ECD (μm)',
       'Length (μm)',  'Perimeter (μm)', 'Shape','Sphericity']]
fig, axPhy = plt.subplots(1,1,figsize=(10,10))
sns.boxplot(data=Phy,ax=axPhy)
axPhy.set_xticklabels(axPhy.get_xticklabels(),rotation=20)
axPhy.set(ylim=(0, 2))
fig.suptitle(file, fontsize=16)
fig.savefig('Physical Properties,'+file+'.jpg')

df.to_csv('Physical Properties'+file[:-3]+'.csv')

df['ArEqSi']=df['Si (Wt%)']*df['Area (μm²)']
df['ArEqS']=df['S (Wt%)']*df['Area (μm²)']
dfSum=df.sum()

print('Area weighted S content %='+str(dfSum['ArEqS']*100/dfSum['Area (μm²)']))
print('Area weighted Si content %='+str(dfSum['ArEqSi']*100/dfSum['Area (μm²)']))


# Load NaCl aerosol data
nacl_data = df

# Load SMPS data
smps_data = pd.read_csv(smps_data_path)



try:
    bin_edges_smps = smps_data['Unnamed: 0'].values
except:
    bin_edges_smps = smps_data['Sample #'].values

# nm to um
bin_edges_smps = bin_edges_smps / 1000
concentration_smps = smps_data['0'].values

# Histogram for the NaCl data
if 'ECD (μm)' in nacl_data.columns:
    ecd_data = nacl_data['ECD (μm)'].dropna()
    min_ecd = ecd_data.min()
    max_ecd = ecd_data.max()
    log_min_ecd = np.log10(min_ecd)
    log_max_ecd = np.log10(max_ecd)
    log_bin_width = 0.037
    log_bin_edges = np.arange(log_min_ecd, log_max_ecd + log_bin_width, log_bin_width)
    bin_edges_log = 10 ** log_bin_edges
    counts, _ = np.histogram(ecd_data, bins=bin_edges_log)
    log_bin_edges = np.log10(bin_edges_log)
    bin_widths = np.diff(bin_edges_log)
    dN_dlogDp = counts / bin_widths
    bin_centers_log = (bin_edges_log[:-1] + bin_edges_log[1:]) / 2
    # Fit log-normal distributions
    from scipy.optimize import curve_fit
    from numpy import exp

    def gaus(x, a, x0, sigma):
        """
        Gaussian function.
        x : Independent variable
        a : Amplitude
        x0: Mean
        sigma: Standard deviation
        """

#        return a * exp(-(np.log10(x-x0))**2 / (2 * sigma**2))
        dp=x
        return (a/(np.sqrt(2*np.pi)*np.log(sigma)*np.log(dp))*\
              np.exp(-(np.log(dp)-np.log(x0))**2/(2*np.log(sigma)**2)))

    # Function to fit the data to a Gaussian distribution
    def fit_gaussian(data_x, data_y):
        """
        Fit the provided data to a Gaussian distribution.
        data_x: The x values of the data
        data_y: The y values of the data
        """
        # Initial guesses for a, x0, and sigma
        initial_guesses = [max(data_y), np.mean(data_x), np.std(data_x)]
        params, _ = curve_fit(gaus, data_x, data_y, p0=initial_guesses)
        return params
    
    bin_centers = (bin_edges_log[:-1] + bin_edges_log[1:]) / 2
    # Fit the SMPS and CCSEM data to the Gaussian function

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Particle Diameter (μm)')
    ax1.set_ylabel('SMPS dN/dlogDp', color='tab:blue')
    ax1.step(bin_edges_smps, concentration_smps, where='mid', color='tab:blue', label='SMPS Data', linewidth=2)
 
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('CCSEM dN/dlogDp', color='tab:red')
    ax2.bar(bin_edges_log[:-1], dN_dlogDp, width=bin_widths, align='edge', edgecolor='black', color='red', alpha=0.5)

    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(bottom=0)


    
    ax1.set_xscale('log')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    try:
        params_smps = fit_gaussian(bin_edges_smps, concentration_smps)
        params_nacl = fit_gaussian(bin_centers_log, dN_dlogDp)
        ax1.plot(bin_edges_smps, gaus(bin_edges_smps, *params_smps), color='tab:blue', linestyle='--', label='SMPS Gaussian Fit')
        ax2.plot(bin_centers, gaus(bin_centers, *params_nacl), color='tab:red', linestyle='--', label='CCSEM Gaussian Fit')

        a, x0, sigma = params_smps
        textstr = f'SMPS Fit Parameters:\na = {a:.4f}\nx0 = {x0:.4f} μm\nσ = {sigma:.2f}'
        ax1.text(0.02, 0.85, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5, color='blue'))

        a, x0, sigma = params_nacl
        textstr = f'CCSEM Fit Parameters:\na = {a:.4f}\nx0 = {x0:.4f} μm\nσ = {sigma:.2f}'
        ax1.text(0.02, 0.45, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5, color='red'))
    except:
        print('No fit was acheived')
    import os
    path=os.getcwd()
    folder=os.path.basename(path)

    plt.title('Size dist Comp,'+file[:-3])
    fig.savefig('Distribution Comparison'+folder+'.png')
    
    
    ccsem_distribution = pd.DataFrame({'bin_centers_log': bin_edges_log[:-1], 'dN/dlogDp': dN_dlogDp})
    ccsem_distribution.to_csv('CCSEM dNdlogDp dist'+file[:-3]+'.csv')
    

    
    plt.show()
else:
    print("ECD (μm) column not found in the dataset.")




plt.show()
