#------------------------import, scrub, wrangle ---------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

df = pd.read_csv("FW_Veg_Rem_Combined.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#drop columns that are not needed. Was not sure if non needed columns maybe had FaF in rows that had all other usable values in needed columns for analysis, so wanted to drop before doing a FaF sweep

# df.drop('disc_clean_date', inplace=True, axis=1) # for some reason this column was already removed
df.drop('Unnamed: 0.1', inplace=True, axis=1)
df.drop('Unnamed: 0', inplace=True, axis=1)
df.drop('fire_name', inplace=True, axis=1)
df.drop('fire_size_class', inplace=True, axis=1)
df.drop('stat_cause_descr', inplace=True, axis=1)
df.drop('state', inplace=True, axis=1)
df.drop('cont_clean_date', inplace=True, axis=1)
df.drop('discovery_month', inplace=True, axis=1)
df.drop('disc_date_final', inplace=True, axis=1)
df.drop('cont_date_final', inplace=True, axis=1)
df.drop('putout_time', inplace=True, axis=1)
df.drop('disc_date_pre', inplace=True, axis=1)
df.drop('disc_pre_year', inplace=True, axis=1)
df.drop('disc_pre_month', inplace=True, axis=1)
df.drop('wstation_usaf', inplace=True, axis=1)
df.drop('dstation_m', inplace=True, axis=1)
df.drop('wstation_wban', inplace=True, axis=1)
df.drop('wstation_byear', inplace=True, axis=1)
df.drop('wstation_eyear', inplace=True, axis=1)
df.drop('Vegetation', inplace=True, axis=1)
df.drop('weather_file', inplace=True, axis=1)
df.drop('disc_clean_date', inplace=True, axis=1)
df.drop('latitude', inplace=True, axis=1)
df.drop('longitude', inplace=True, axis =1)
# print(df.head())

# more scrubbing
print("How many FaF values do we Have:", df.isnull().sum().sum()) #do we have FaF?
print("do we have negative values:",(df < 0).any().any()) # do we have negative values ?

# now remove negative values -- looks like -1 was assigned to rows as a signifier some type of sensor error? Feels appropriate to
#remove any negative values as U couldn't understand what a negative value would physically mean in the context of our domain values
col_names = list(df.columns)
for name in col_names:
    df = df.drop(df.index[df[name] <= 0])



print("do we have negative values:", (df <= 0).any().any()) #check to make sure we got all negative values
#print(df.head())

#pull values of interest
fire_size = df['fire_size'].values
fire_mag = df['fire_mag'].values
temp_30 = df['Temp_pre_30'].values
temp_15 = df['Temp_pre_15'].values
temp_7 = df['Temp_pre_7'].values
temp_dur = df['Temp_cont'].values
wind_30 = df['Wind_pre_30'].values
wind_15 = df['Wind_pre_15'].values
wind_7 = df['Wind_pre_7'].values
wind_dur = df['Wind_cont'].values
hum_30 = df['Hum_pre_30'].values
hum_15 = df['Hum_pre_15'].values
hum_7 = df['Hum_pre_7'].values
hum_dur = df['Hum_cont'].values
prec_30 = df['Prec_pre_30'].values
prec_15 = df['Prec_pre_15'].values
prec_7 = df['Prec_pre_7'].values
prec_dur = df['Prec_cont'].values
remotness = df['remoteness'].values

#---------------------------------Regression-------------------------------------------

y_obs = fire_mag

def model(parameters):
    a = parameters[0]; b= parameters[1]; c = parameters[2]; d = parameters[3]
    e = parameters[4]; f= parameters[5]; g = parameters[6]; h = parameters[7]
    i = parameters[8]; j= parameters[9]; k=parameters[10]
   
    y_hat = b*temp_30**a + \
           d*wind_30**c +\
           f*hum_30**e +\
           h*prec_30**g +\
           i*remotness**j + k
    return y_hat

def model_error(params):
    # print("these are param", params)
    y_hat = model(params)
    MSE = mean_squared_error(y_obs, y_hat)
    BMSE = math.sqrt(MSE)
    return BMSE
    #return np.sum( (y_obs - y_hat)**2 ) # sum of squares error (SSE)
    

#initial guess 
p0 = []
for i in range(0,11):
    p0.append(1)
    

res = minimize(model_error,p0)
print(res)
def get_r_value(obsin,modelin):
    
    return (1 - np.sum((obsin - modelin)**2)/np.sum((obsin - np.mean(obsin))**2))**0.5

ymodeled3 = model(res.x)
print(np.sum(y_obs-ymodeled3))

print(y_obs)
print(ymodeled3)

from sklearn.metrics import r2_score

r2 = r2_score(y_obs,ymodeled3)
print(r2)

print('Model 3 r-squared:',get_r_value(y_obs,ymodeled3))
# Make a cross-plot
plt.plot(y_obs,ymodeled3,'bo', alpha=0.1)
plt.plot(y_obs,y_obs, 'k-') # 1:1 line
#plt.axline((0,0),slope=1) # different method for plotting 1:1 line
plt.xlabel('Observations ($y_i$)'); plt.ylabel('Model Estimates ($\hat{y_i}$)')

#----------------------------------------dynamics---------------------------------------------------------
iSize = 45
jSize = 80

# Inital Population Counts, for N can just read CUV data into needed 2d data array in one step
N = np.genfromtxt("BEE529 M6 Dataset NApop.csv", delimiter=',')
I0 = np.zeros((iSize,jSize));
B0 = np.zeros((iSize,jSize));

# define random initialization function to find infected individual start point.
def random_init():
    initial_x = random.randint(0,79) 
    initial_y = random.randint(0,44)
    return [initial_y, initial_x]

# define loop preventing checking init population to make sure infected individual from starting in the ocean
attempt = random_init()
while (N[attempt[0]][attempt[1]] == 0.0 and N[attempt[0]][attempt[1]] <= 5000):
    print("hit")
    attempt = random_init()

I0[attempt[0]][attempt[1]] = 5000 # assign one infected individual 
U0 = N - I0 - B0

# find total population from CUV data
nMax = 0.0
for row in N:
    for value in row:
        nMax+=value

alpha = .001
beta = 0.19 
gamma = .1
print('r_0 is', beta/gamma)
# Migration Bate [people per day of the total population per boundary]
percent_migrate = .05
# A grid of time points (in days)
simulation_days = 5475
dt = 0.1
t = np.linspace(0, simulation_days, int(simulation_days/dt))

# Empty Output Location
yOut = np.zeros((iSize,jSize,len(t),3))

# The UIB model differential equations.
def spatialUIB(y):
    # Current Uystem Utatus
    U = y[0]; I = y[1]; B = y[2]
    # Empty Derivitive Terms
    dUdt = np.copy(U)*0; 
    dIdt = np.copy(I)*0;
    dBdt = np.copy(B)*0;
    
    # loop through all locations
    for i in np.arange(U.shape[0]):
        for j in np.arange(U.shape[1]):
        
            if N[i,j] != 0: # check not on ocean cell
            
                # internal contribution
                dUdt[i,j] = ((-beta*U[i,j]*I[i,j]/N[i,j]) + alpha*B[i,j])
                dIdt[i,j] = (beta*U[i,j]*I[i,j]/N[i,j] - gamma*I[i,j])
                dBdt[i,j] = (gamma*I[i,j]-alpha*B[i,j])

                # find valid boundaries and calculate change contribution of immigration and emigration
                if i>0 and N[i-1,j] != 0: # north check
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i-1,j]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i-1,j]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
                    
                    dIdt[i,j] += mrate*(I[i-1,j]/N[i-1,j])
                    

                    #emigration technique
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
                    

                if j>0 and N[i,j-1] != 0: #west check 
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i,j-1]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i,j-1]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
        
                    dIdt[i,j] += mrate*(I[i,j-1]/N[i,j-1])
                

                    #emigration technique
            
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
           

                if i<44 and N[i+1,j] != 0: #south check 
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i+1,j]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i+1,j]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
                    
                    dIdt[i,j] += mrate*(I[i+1,j]/N[i+1,j])
                    

                    #emigration technique
                    
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
                          

                if j<79 and N[i,j+1] != 0: #east check 
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i,j+1]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i,j+1]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
                    
                    dIdt[i,j] += mrate*(I[i,j+1]/N[i,j+1])
                    

                    #emigration technique
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
                     

    # Get next value
    Uout = U + dUdt*dt
    Iout = I + dIdt*dt
    Bout = B + dBdt*dt
    return [Uout, Iout, Bout]

# Uet initial Conditions
yOut[:,:,0,0] = U0; yOut[:,:,0,1] = I0; yOut[:,:,0,2] = B0;

# Iterate into future
for curt in np.arange(1,len(t)):
    y0 = [yOut[:,:,curt-1,0], yOut[:,:,curt-1,1], yOut[:,:,curt-1,2]]
    [yOut[:,:,curt,0], yOut[:,:,curt,1], yOut[:,:,curt,2]] = spatialUIB(y0)

# ------------------------------------plots for dynamics-------------------------------------------
print(attempt[0],attempt[1])
print(N[attempt[0],attempt[1]])

# initialization cell 
plt.figure(1)
U1 = yOut[attempt[0],attempt[1],:,0]; F1 = yOut[attempt[0],attempt[1],:,1]; B1 = yOut[attempt[0],attempt[1],:,2]
plt.plot(t, U1/1000, 'g', label='Unburnt')
plt.plot(t, F1/1000, 'r', label='Active Burning')
plt.plot(t, B1/1000, 'k', label='Burnt')
plt.xlabel('Time /days')
plt.ylabel('Number (1000s)')
plt.legend(loc='center right')
print(F1)

# adjacent cell
plt.figure(2)
U1 = yOut[attempt[0]+1,attempt[1]+1,:,0]; F1 = yOut[attempt[0]+1,attempt[1]+1,:,1]; B1 = yOut[attempt[0]+1,attempt[1]+1,:,2]
plt.plot(t, U1/1000, 'g', label='Unburnt')
plt.plot(t, F1/1000, 'r', label='Active Burning')
plt.plot(t, B1/1000, 'k', label='Burnt')
plt.xlabel('Time /days')
plt.xlim(0,500)
plt.ylabel('Number (1000s)')
plt.legend(loc='center right')

#animation
# Uet Up Axes
from matplotlib import animation
fig, axs = plt.subplots(1, 3,figsize=(12, 3))
cax0 = axs[0].pcolormesh(np.log(np.flipud(yOut[:, :, 0, 0])))
cax1 = axs[1].pcolormesh(np.log(np.flipud(yOut[:, :, 0, 1])),vmin=0, vmax=1.)
cax2 = axs[2].pcolormesh(np.log(np.flipud(yOut[:, :, 0, 2])),vmin=0, vmax=1.)
fig.colorbar(cax2)
fig.colorbar(cax1)
fig.colorbar(cax0)

# What to Plot at i
def animate(i):
    cax0.set_array(np.log(np.flipud(yOut[:, :, i,0])))
    cax1.set_array(np.log(np.flipud(yOut[:, :, i,1])))
    cax2.set_array(np.log(np.flipud(yOut[:, :, i,2])))
    axs[0].set_title('U at %d days' %t[i])
    axs[1].set_title('F at %d days' %t[i])
    axs[2].set_title('B at %d days' %t[i])
    plt.tight_layout()

# Make the Animation
anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(t), 50), interval = 10)
gif_name = 'final_burn_test7.gif'
anim.save(gif_name)

# Display the GFF
from IPython.display import Image
Image(url=gif_name) 

#-----------------------------sensitivity analysis---------------------------
# Total population, N.
print(N[attempt[0]][attempt[1]])
Ncopy = N[attempt[0]][attempt[1]]
# Initial number of infected and recovered individuals, I0 and R0.
F0 = 5000; B0 = 0
# Everyone else, S0, is susceptible to infection initially.
U0 = Ncopy - F0 - B0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
sens_alpha = .001
sens_beta = 0.15
sens_gamma = .2

print('r_0 is', beta/gamma)
# A grid of time points (in days)


# The SIR model differential equations.
def SIRmodel(y, t):
    U = y[0]
    F = y[1]
    B = y[2]
    dUdt = -sens_beta*U*F/Ncopy +sens_alpha*B
    dFdt = sens_beta*U*F/Ncopy - sens_gamma*F
    dBdt = sens_gamma*F -sens_alpha*B
    return [dUdt, dFdt, dBdt]

def runTheModel():
    simulation_days = 365
    times = np.linspace(0, simulation_days, simulation_days)
    # Initial conditions vector
    y0 = [U0, F0, B0]
    # Integrate the SIR equations over the time grid, t.
    Y1 = odeint(SIRmodel, y0, t)
    return times, Y1
rescaleValue = 0.1

# store base case parameter values
Burn_base = sens_beta
extinguish_base = sens_gamma
regrow_base = sens_alpha

# run the base case
times,Yba  = runTheModel()

# perturb in burn rate
sens_alpha= Burn_base * (1+rescaleValue)    # perturb by 10 percent
times, Yb = runTheModel()
sens_alpha = Burn_base                                # restore starting value after run

# perturb in extinguish rate
sens_gamma = extinguish_base * (1+rescaleValue)    # perturb by 10 percent
times,Ye = runTheModel()
sens_gamma = extinguish_base    # restore starting value after run

# perturb in regrow rate
sens_alpha = regrow_base * (1+rescaleValue)    # perturb by 10 percent
times,Yr = runTheModel()
sens_alpha = regrow_base      # restore starting value after run

# generate sensitivity table - rows are parameters, columns are sensitivity results
Bab = Yba[-1,0]  # base burn area
print( 'Base burn area (Response Value): {:.3}'.format(Bab))

# burnrate
Bap = Yb[-1,0] # perturbed case
absSensitivity = (Bap-Bab)/(rescaleValue*Burn_base)   
relSensitivity = ((Bap-Bab)/Bab)/rescaleValue
results_burnate = ('Burn rate',Burn_base,Burn_base*(1+rescaleValue),Bap,absSensitivity,relSensitivity)

#  extinguish 
Bap = Ye[-1,0] # perturbed case
absSensitivity = (Bap-Bab)/(rescaleValue*extinguish_base)   
relSensitivity = ((Bap-Bab)/Bab)/rescaleValue
results_extinguish = ('Extinguish',extinguish_base,extinguish_base*(1+rescaleValue),Bap,absSensitivity,relSensitivity)

# regrow
Bap = Yr[-1,0] # perturbed case
absSensitivity = (Bap-Bab)/(rescaleValue*regrow_base)   
relSensitivity = ((Bap-Bab)/Bab)/rescaleValue
results_regrow = ('Regrow',regrow_base,regrow_base*(1+rescaleValue),Bap,absSensitivity,relSensitivity)

# make a dataframe to hold the results
records = [ results_burnate, results_extinguish, results_regrow]
labels = ['Parameter', 'Base Value', 'Perturbed Value', 'Response Value', 'Absolute Sensitivity', 'Relative Sensitivity']
df = pd.DataFrame.from_records(records, columns=labels)

rescaleFactor = 2
rescaleNumbers = 100

# perturb sense_beta burn rate parameter
sens_beta_list = np.linspace(sens_beta/rescaleFactor,sens_beta*rescaleFactor,rescaleNumbers)
B_out_list = np.copy(sens_beta_list)
# Run for each value
for i in np.arange(len(sens_beta_list)):
    sens_beta = sens_beta_list[i]
    times, Yfr = runTheModel()
    B_out_list[i] = Yfr[-1,0]
sens_beta = Burn_base

# Plot the local sensitivity
plt.figure(1)
relSensitivity = np.diff(B_out_list)/np.diff(sens_beta_list)
relSensitivity = np.append(relSensitivity,np.nan)
plt.plot(sens_beta_list,relSensitivity,'-')
plt.xlabel('Beta (burn rate)'); plt.ylabel('Local Sensitivity')

# perturb sense_gamma extinguish parameter
sens_gamma_list = np.linspace(sens_gamma/rescaleFactor,sens_gamma*rescaleFactor,rescaleNumbers)
B_out_list = np.copy(sens_gamma_list)
# Run for each value
for i in np.arange(len(sens_gamma_list)):
    sens_gamma = sens_gamma_list[i]
    times, Yfr = runTheModel()
    B_out_list[i] = Yfr[-1,0]
sens_gamma = extinguish_base

plt.figure(2)
relSensitivity = np.diff(B_out_list)/np.diff(sens_gamma_list)
relSensitivity = np.append(relSensitivity,np.nan)
plt.plot(sens_gamma_list,relSensitivity,'-')
plt.xlabel('gamma (extinguish rate)'); plt.ylabel('Local Sensitivity')


# perturb sense_alpha parameter
sens_alpha_list = np.linspace(sens_alpha/rescaleFactor,sens_alpha*rescaleFactor,rescaleNumbers)
B_out_list = np.copy(sens_alpha_list)
# Run for each value
for i in np.arange(len(sens_alpha_list)):
    sens_alpha = sens_alpha_list[i]
    times, Yfr = runTheModel()
    B_out_list[i] = Yfr[-1,0]
sens_alpha = regrow_base

# Plot the local sensitivity
plt.figure(3)
relSensitivity = np.diff(B_out_list)/np.diff(sens_alpha_list)
relSensitivity = np.append(relSensitivity,np.nan)
plt.plot(sens_alpha_list,relSensitivity,'-')
plt.xlabel('Alpha (regrow rate)'); plt.ylabel('Local Sensitivity')

# And now for both
sens_beta_mat, sens_gamma_mat = np.meshgrid(sens_beta_list, sens_gamma_list, sparse=False, indexing='ij')
f_out_mat = np.zeros((len(sens_beta_list),len(sens_gamma_list)))
for i in np.arange(len(sens_beta_list)):
    for j in np.arange(len(sens_gamma_list)):
        sens_beta = sens_beta_mat[i,j]
        sens_gamma = sens_gamma_mat[j,j]
        times, Yfr = runTheModel()
        f_out_mat[i,j] = Yfr[-1,0]
sens_beta = Burn_base
sens_gamma = extinguish_base

# Plot the f out
plt.figure(1)
plt.contourf(sens_beta_mat,sens_gamma_mat,f_out_mat)
plt.xlabel('Beta'); plt.ylabel('Gamma')
plt.colorbar(label='Burn Area')

# Take the local gradient
grad = np.gradient(f_out_mat)

plt.figure(2,figsize=(14,4))
plt.subplot(1,2,1)
plt.contourf(sens_beta_mat,sens_gamma_mat,grad[0])
plt.xlabel('Beta'); plt.ylabel('Gamma')
plt.colorbar(label='Sensitivity to Beta')
plt.figure(2)
plt.subplot(1,2,2)
plt.contourf(sens_beta_mat,sens_gamma_mat,grad[1])
plt.xlabel('Beta'); plt.ylabel('Gamma')
plt.colorbar(label='Sensitivity to Gamma')