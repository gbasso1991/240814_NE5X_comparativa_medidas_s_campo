#%% Analisis para calculo de tau - levantando de archivo resultados.txt y de ciclo
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy 
from scipy.optimize import curve_fit 
from scipy.stats import linregress
from uncertainties import ufloat, unumpy
from glob import glob
#%% LECTOR RESULTADOS
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(6):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=14,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.to_datetime(data['Time_m'][:],dayfirst=True)
    # delta_t = np.array([dt.total_seconds() for dt in (time-time[0])])
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
     
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

def lector_resultados_2(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(6):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=19,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    # delta_t = np.array([dt.total_seconds() for dt in (time-time[0])])
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
     
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:6]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t= pd.Series(data['Tiempo_(s)']).to_numpy()
    H = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M= pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H,M,metadata

#%% TAU PROMEDIO
def Tau_promedio(filepath,recorto_extremos=20):
    '''Dado un path, toma archivo de ciclo M vs H
     Calcula Magnetizacion de Equilibrio, y Tau pesado con dM/dH
     '''
    t,H,M,meta=lector_ciclos(filepath)
     
    indx_max= np.nonzero(H==max(H))[0][0]
    t_mag = t[recorto_extremos:indx_max-recorto_extremos]
    H_mag = H[recorto_extremos:indx_max-recorto_extremos]
    M_mag = M[recorto_extremos:indx_max-recorto_extremos]

    H_demag = H[indx_max+recorto_extremos:-recorto_extremos] 
    # H_demag = np.concatenate((H_demag[:],H_mag[0:1]))

    M_demag = M[indx_max+recorto_extremos:-recorto_extremos]
    # M_demag = np.concatenate((M_demag[:],M_mag[0:1]))

    #INTERPOLACION de M 
    # Verificar que H_mag esté dentro del rango de H_demag
    #H_mag = H_mag[(H_mag >= min(H_demag)) & (H_mag <= max(H_demag))]

    # INTERPOLACION de M solo para los valores dentro del rango
    interpolador = interp1d(H_demag, M_demag,fill_value="extrapolate")
    M_demag_int = interpolador(H_mag)

    # interpolador=interp1d(H_demag, M_demag)
    # M_demag_int = interpolador(H_mag) 
    
    # Derivadas
    dMdH_mag = np.gradient(M_mag,H_mag)
    dMdH_demag_int = np.gradient(M_demag_int,H_mag)
    dHdt= np.gradient(H_mag,t_mag)

    Meq = (M_mag*dMdH_demag_int + M_demag_int*dMdH_mag)/(dMdH_mag+ dMdH_demag_int)
    dMeqdH = np.gradient(Meq,H_mag)

    Tau = (Meq - M_mag)/(dMdH_mag*dHdt )

    Tau_prom = np.sum(Tau*dMeqdH)/np.sum(dMdH_mag)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #%paso a kA/m y ns
    H_mag/=1e3
    H_demag/=1e3
    Tau *=1e9
    Tau_prom*=1e9
    print(meta['filename'])
    print(Tau_prom,'s')

    fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(7,6),constrained_layout=True)
    #ax1.plot(H,Tau,'-',label='U')
    ax1.plot(H_mag,Tau,'.-')
    ax1.grid()
    ax1.set_xlabel('H (kA/m)')
    ax1.set_ylabel(r'$\tau$ (s)')
    ax1.text(1/2,1/7,rf'<$\tau$> = {Tau_prom:.1f} ns',ha='center',va='center',
             bbox=dict(alpha=0.8),transform=ax1.transAxes,fontsize=11)

    ax1.grid()
    ax1.set_xlabel('H (A/m)')
    ax1.set_ylabel('$\\tau$ (ns)')
    ax1.set_title(r'$\tau$ vs H', loc='left')
    ax1.grid()

    ax2.plot(H_mag,Meq,'-',label='M$_{equilibrio}$')
    ax2.plot(H_mag,M_mag,label='Mag')
    ax2.plot(H_demag,M_demag,label='Demag')
    ax2.grid()
    ax2.legend()
    ax2.set_title('M vs H', loc='left')
    ax2.set_xlabel('H (kA/m)')
    ax2.set_ylabel('M (A/m)')

    axins = ax2.inset_axes([0.6, 0.12, 0.39, 0.4])
    axins.plot(H_mag,Meq,'.-')
    axins.plot(H_mag, M_mag,'.-')
    axins.plot(H_demag,M_demag,'.-')
    axins.set_xlim(-0.1*max(H_mag),0.1*max(H_mag)) 
    axins.set_ylim(-0.1*max(M_mag),0.1*max(M_mag))
    ax2.indicate_inset_zoom(axins, edgecolor="black")
    axins.grid()
    plt.suptitle(meta['filename'])

    return Meq , H_mag, max(H)/1000, Tau , Tau_prom , fig

#%% TEMPLOG
from datetime import datetime

def lector_templog(directorio, plot=False):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura y plotea el log completo 
    '''
    if fnmatch.filter(os.listdir(directorio),'*templog*'):
        dir_templog = os.path.join(directorio,fnmatch.filter(os.listdir(directorio),'*templog*')[0])#toma el 1ero en orden alfabetico
        data = pd.read_csv(dir_templog,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
        
        temp_CH1 = pd.Series(data['T_CH1']).to_numpy(dtype=float)
        temp_CH2= pd.Series(data['T_CH2']).to_numpy(dtype=float)
        timestamp=np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
        if plot:
            pass
            # fig, ax = plt.subplots(figsize=(10,5))
            # ax.plot(timestamp,temp_CH1,'.-',label=dir_templog.split('_')[-1]+' CH1' )
            # ax.plot(timestamp,temp_CH2,'.-',label=dir_templog.split('_')[-1]+ ' CH2')
            # ax.plot(timestamp,np.abs(temp_CH1-temp_CH2),'.-',label=rf'$\Delta$ max = {max(np.abs(temp_CH1-temp_CH2)):.1f} °C')
            # plt.grid()
            # plt.ylabel('Temperatura (ºC)')
            # fig.autofmt_xdate()
            # plt.legend(loc='best')  
            # plt.tight_layout()
            # plt.xlim(timestamp[0],timestamp[-1])
            # plt.show()
        else:
            pass
        return timestamp,temp_CH1, temp_CH2
    else:
        print('No se encuentra archivo templog.csv en el directorio:',directorio)


#%% (a) vs (c): 135 kHz  descong sin campo DC - distintas fechas  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_a= 'a  - 26-Jun - 135 kHz'
files_a_150 = glob(os.path.join(dir_a, '*150dA*'))
files_a_100 = glob(os.path.join(dir_a, '*100dA*'))
files_a_050 = glob(os.path.join(dir_a, '*050dA*'))

_,files_a_150_1,time_a_150_1,temp_a_150_1,_,_,_,_,_,_,_,_,SAR_a_150_1,tau_a_150_1,_ = lector_resultados(files_a_150[0])
_,files_a_150_2,time_a_150_2,temp_a_150_2,_,_,_,_,_,_,_,_,SAR_a_150_2,tau_a_150_2,_ = lector_resultados(files_a_150[1])
_,files_a_150_3,time_a_150_3,temp_a_150_3,_,_,_,_,_,_,_,_,SAR_a_150_3,tau_a_150_3,_ = lector_resultados(files_a_150[2])

_,files_a_100_1,time_a_100_1,temp_a_100_1,_,_,_,_,_,_,_,_,SAR_a_100_1,tau_a_100_1,_ = lector_resultados(files_a_100[0])
_,files_a_100_2,time_a_100_2,temp_a_100_2,_,_,_,_,_,_,_,_,SAR_a_100_2,tau_a_100_2,_ = lector_resultados(files_a_100[1])
_,files_a_100_3,time_a_100_3,temp_a_100_3,_,_,_,_,_,_,_,_,SAR_a_100_3,tau_a_100_3,_ = lector_resultados(files_a_100[2])

_,files_a_050_1,time_a_050_1,temp_a_050_1,_,_,_,_,_,_,_,_,SAR_a_050_1,tau_a_050_1,_ = lector_resultados(files_a_050[0])
_,files_a_050_2,time_a_050_2,temp_a_050_2,_,_,_,_,_,_,_,_,SAR_a_050_2,tau_a_050_2,_ = lector_resultados(files_a_050[1])
_,files_a_050_3,time_a_050_3,temp_a_050_3,_,_,_,_,_,_,_,_,SAR_a_050_3,tau_a_050_3,_ = lector_resultados(files_a_050[2])

dir_c= 'c - 18-Jul - 135 kHz'
files_c_150 = glob(os.path.join(dir_c, '*150dA*'))
files_c_100 = glob(os.path.join(dir_c, '*100dA*'))
files_c_050 = glob(os.path.join(dir_c, '*050dA*'))

_,files_c_150_1,time_c_150_1,temp_c_150_1,_,_,_,_,_,_,_,_,SAR_c_150_1,tau_c_150_1,_ = lector_resultados_2(files_c_150[0])
_,files_c_150_2,time_c_150_2,temp_c_150_2,_,_,_,_,_,_,_,_,SAR_c_150_2,tau_c_150_2,_ = lector_resultados_2(files_c_150[1])
_,files_c_150_3,time_c_150_3,temp_c_150_3,_,_,_,_,_,_,_,_,SAR_c_150_3,tau_c_150_3,_ = lector_resultados_2(files_c_150[2])

_,files_c_100_1,time_c_100_1,temp_c_100_1,_,_,_,_,_,_,_,_,SAR_c_100_1,tau_c_100_1,_ = lector_resultados_2(files_c_100[0])
_,files_c_100_2,time_c_100_2,temp_c_100_2,_,_,_,_,_,_,_,_,SAR_c_100_2,tau_c_100_2,_ = lector_resultados_2(files_c_100[1])
_,files_c_100_3,time_c_100_3,temp_c_100_3,_,_,_,_,_,_,_,_,SAR_c_100_3,tau_c_100_3,_ = lector_resultados_2(files_c_100[2])

_,files_c_050_1,time_c_050_1,temp_c_050_1,_,_,_,_,_,_,_,_,SAR_c_050_1,tau_c_050_1,_ = lector_resultados_2(files_c_050[0])
_,files_c_050_2,time_c_050_2,temp_c_050_2,_,_,_,_,_,_,_,_,SAR_c_050_2,tau_c_050_2,_ = lector_resultados_2(files_c_050[1])
_,files_c_050_3,time_c_050_3,temp_c_050_3,_,_,_,_,_,_,_,_,SAR_c_050_3,tau_c_050_3,_ = lector_resultados_2(files_c_050[2])

#% paso tau a ns
tau_a_150_1=tau_a_150_1*1e9
tau_a_150_2=tau_a_150_2*1e9
tau_a_150_3=tau_a_150_3*1e9
tau_a_100_1=tau_a_100_1*1e9
tau_a_100_2=tau_a_100_2*1e9
tau_a_100_3=tau_a_100_3*1e9
tau_a_050_1=tau_a_050_1*1e9
tau_a_050_2=tau_a_050_2*1e9
tau_a_050_3=tau_a_050_3*1e9

tau_c_150_1=tau_c_150_1*1e9
tau_c_150_2=tau_c_150_2*1e9
tau_c_150_3=tau_c_150_3*1e9
tau_c_100_1=tau_c_100_1*1e9
tau_c_100_2=tau_c_100_2*1e9
tau_c_100_3=tau_c_100_3*1e9
tau_c_050_1=tau_c_050_1*1e9
tau_c_050_2=tau_c_050_2*1e9
tau_c_050_3=tau_c_050_3*1e9
#%encuentro los maximos 
tau_max_a_150_1=tau_a_150_1[np.nonzero(tau_a_150_1==max(tau_a_150_1))][0]
tau_max_a_150_2=tau_a_150_2[np.nonzero(tau_a_150_2==max(tau_a_150_2))][0]
tau_max_a_150_3=tau_a_150_3[np.nonzero(tau_a_150_3==max(tau_a_150_3))][0]
tau_max_a_100_1=tau_a_100_1[np.nonzero(tau_a_100_1==max(tau_a_100_1))][0]
tau_max_a_100_2=tau_a_100_2[np.nonzero(tau_a_100_2==max(tau_a_100_2))][0]
tau_max_a_100_3=tau_a_100_3[np.nonzero(tau_a_100_3==max(tau_a_100_3))][0]
tau_max_a_050_1=tau_a_050_1[np.nonzero(tau_a_050_1==max(tau_a_050_1))][0]
tau_max_a_050_2=tau_a_050_2[np.nonzero(tau_a_050_2==max(tau_a_050_2))][0]
tau_max_a_050_3=tau_a_050_3[np.nonzero(tau_a_050_3==max(tau_a_050_3))][0]

temp_max_a_150_1=temp_a_150_1[np.nonzero(tau_a_150_1==max(tau_a_150_1))][0]
temp_max_a_150_2=temp_a_150_2[np.nonzero(tau_a_150_2==max(tau_a_150_2))][0]
temp_max_a_150_3=temp_a_150_3[np.nonzero(tau_a_150_3==max(tau_a_150_3))][0]
temp_max_a_100_1=temp_a_100_1[np.nonzero(tau_a_100_1==max(tau_a_100_1))][0]
temp_max_a_100_2=temp_a_100_2[np.nonzero(tau_a_100_2==max(tau_a_100_2))][0]
temp_max_a_100_3=temp_a_100_3[np.nonzero(tau_a_100_3==max(tau_a_100_3))][0]
temp_max_a_050_1=temp_a_050_1[np.nonzero(tau_a_050_1==max(tau_a_050_1))][0]
temp_max_a_050_2=temp_a_050_2[np.nonzero(tau_a_050_2==max(tau_a_050_2))][0]
temp_max_a_050_3=temp_a_050_3[np.nonzero(tau_a_050_3==max(tau_a_050_3))][0]

tau_max_c_150_1=tau_c_150_1[np.nonzero(tau_c_150_1==max(tau_c_150_1))][0]
tau_max_c_150_2=tau_c_150_2[np.nonzero(tau_c_150_2==max(tau_c_150_2))][0]
tau_max_c_150_3=tau_c_150_3[np.nonzero(tau_c_150_3==max(tau_c_150_3))][0]
tau_max_c_100_1=tau_c_100_1[np.nonzero(tau_c_100_1==max(tau_c_100_1))][0]
tau_max_c_100_2=tau_c_100_2[np.nonzero(tau_c_100_2==max(tau_c_100_2))][0]
tau_max_c_100_3=tau_c_100_3[np.nonzero(tau_c_100_3==max(tau_c_100_3))][0]
tau_max_c_050_1=tau_c_050_1[np.nonzero(tau_c_050_1==max(tau_c_050_1))][0]
tau_max_c_050_2=tau_c_050_2[np.nonzero(tau_c_050_2==max(tau_c_050_2))][0]
tau_max_c_050_3=tau_c_050_3[np.nonzero(tau_c_050_3==max(tau_c_050_3))][0]

temp_max_c_150_1=temp_c_150_1[np.nonzero(tau_c_150_1==max(tau_c_150_1))][0]
temp_max_c_150_2=temp_c_150_2[np.nonzero(tau_c_150_2==max(tau_c_150_2))][0]
temp_max_c_150_3=temp_c_150_3[np.nonzero(tau_c_150_3==max(tau_c_150_3))][0]
temp_max_c_100_1=temp_c_100_1[np.nonzero(tau_c_100_1==max(tau_c_100_1))][0]
temp_max_c_100_2=temp_c_100_2[np.nonzero(tau_c_100_2==max(tau_c_100_2))][0]
temp_max_c_100_3=temp_c_100_3[np.nonzero(tau_c_100_3==max(tau_c_100_3))][0]
temp_max_c_050_1=temp_c_050_1[np.nonzero(tau_c_050_1==max(tau_c_050_1))][0]
temp_max_c_050_2=temp_c_050_2[np.nonzero(tau_c_050_2==max(tau_c_050_2))][0]
temp_max_c_050_3=temp_c_050_3[np.nonzero(tau_c_050_3==max(tau_c_050_3))][0]

#%% Tau vs T
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(temp_a_150_1,tau_a_150_1)
ax[0,0].plot(temp_a_150_2,tau_a_150_2)
ax[0,0].plot(temp_a_150_3,tau_a_150_3)
ax[0,0].scatter(temp_max_a_150_1,tau_max_a_150_1,marker='v',label=f'{tau_max_a_150_1:.1f} ns at {temp_max_a_150_1}°C')
ax[0,0].scatter(temp_max_a_150_2,tau_max_a_150_2,marker='v',label=f'{tau_max_a_150_2:.1f} ns at {temp_max_a_150_2}°C')
ax[0,0].scatter(temp_max_a_150_3,tau_max_a_150_3,marker='v',label=f'{tau_max_a_150_3:.1f} ns at {temp_max_a_150_3}°C')

ax[0,1].plot(temp_c_150_1,tau_c_150_1)
ax[0,1].plot(temp_c_150_2,tau_c_150_2)
ax[0,1].plot(temp_c_150_3,tau_c_150_3)
ax[0,1].scatter(temp_max_c_150_1,tau_max_c_150_1,marker='v',label=f'{tau_max_c_150_1:.1f} ns at {temp_max_c_150_1}°C')
ax[0,1].scatter(temp_max_c_150_2,tau_max_c_150_2,marker='v',label=f'{tau_max_c_150_2:.1f} ns at {temp_max_c_150_2}°C')
ax[0,1].scatter(temp_max_c_150_3,tau_max_c_150_3,marker='v',label=f'{tau_max_c_150_3:.1f} ns at {temp_max_c_150_3}°C')

ax[1,0].plot(temp_a_100_1,tau_a_100_1)
ax[1,0].plot(temp_a_100_2,tau_a_100_2)
ax[1,0].plot(temp_a_100_3,tau_a_100_3)
ax[1,0].scatter(temp_max_a_100_1,tau_max_a_100_1,marker='v',label=f'{tau_max_a_100_1:.1f} ns at {temp_max_a_100_1}°C')
ax[1,0].scatter(temp_max_a_100_2,tau_max_a_100_2,marker='v',label=f'{tau_max_a_100_2:.1f} ns at {temp_max_a_100_2}°C')
ax[1,0].scatter(temp_max_a_100_3,tau_max_a_100_3,marker='v',label=f'{tau_max_a_100_3:.1f} ns at {temp_max_a_100_3}°C')

ax[1,1].plot(temp_c_100_1,tau_c_100_1)
ax[1,1].plot(temp_c_100_2,tau_c_100_2)
ax[1,1].plot(temp_c_100_3,tau_c_100_3)
ax[1,1].scatter(temp_max_c_100_1,tau_max_c_100_1,marker='v',label=f'{tau_max_c_100_1:.1f} ns at {temp_max_c_100_1}°C')
ax[1,1].scatter(temp_max_c_100_2,tau_max_c_100_2,marker='v',label=f'{tau_max_c_100_2:.1f} ns at {temp_max_c_100_2}°C')
ax[1,1].scatter(temp_max_c_100_3,tau_max_c_100_3,marker='v',label=f'{tau_max_c_100_3:.1f} ns at {temp_max_c_100_3}°C')

ax[2,0].plot(temp_a_050_1,tau_a_050_1,)
ax[2,0].plot(temp_a_050_2,tau_a_050_2,)
ax[2,0].plot(temp_a_050_3,tau_a_050_3,)
ax[2,0].scatter(temp_max_a_050_1,tau_max_a_050_1,marker='v',label=f'{tau_max_a_050_1:.1f} ns at {temp_max_a_050_1}°C')
ax[2,0].scatter(temp_max_a_050_2,tau_max_a_050_2,marker='v',label=f'{tau_max_a_050_2:.1f} ns at {temp_max_a_050_2}°C')
ax[2,0].scatter(temp_max_a_050_3,tau_max_a_050_3,marker='v',label=f'{tau_max_a_050_3:.1f} ns at {temp_max_a_050_3}°C')

ax[2,1].plot(temp_c_050_1,tau_c_050_1,)
ax[2,1].plot(temp_c_050_2,tau_c_050_2,)
ax[2,1].plot(temp_c_050_3,tau_c_050_3,)
ax[2,1].scatter(temp_max_c_050_1,tau_max_c_050_1,marker='v',label=f'{tau_max_c_050_1:.1f} ns at {temp_max_c_050_1}°C')
ax[2,1].scatter(temp_max_c_050_2,tau_max_c_050_2,marker='v',label=f'{tau_max_c_050_2:.1f} ns at {temp_max_c_050_2}°C')
ax[2,1].scatter(temp_max_c_050_3,tau_max_c_050_3,marker='v',label=f'{tau_max_c_050_3:.1f} ns at {temp_max_c_050_3}°C')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=9)
        b.set_xlabel('T (°C)')
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_a, ha='center',fontsize=13)
fig.text(0.775, 0.92,dir_c, ha='center',fontsize=13)
plt.suptitle('NE5X cong s/ $H_{DC}^{\perp}$\n$\\tau$ vs Temperatura',fontsize=16)

#% Tiempos absoultos
time_a_150_1 = np.array([(f-time_a_150_1[0]).total_seconds() for f in  time_a_150_1])
time_a_150_2 = np.array([(f-time_a_150_2[0]).total_seconds() for f in  time_a_150_2])
time_a_150_3 = np.array([(f-time_a_150_3[0]).total_seconds() for f in  time_a_150_3])
time_a_100_1 = np.array([(f-time_a_100_1[0]).total_seconds() for f in  time_a_100_1])
time_a_100_2 = np.array([(f-time_a_100_2[0]).total_seconds() for f in  time_a_100_2])
time_a_100_3 = np.array([(f-time_a_100_3[0]).total_seconds() for f in  time_a_100_3])
time_a_050_1 = np.array([(f-time_a_050_1[0]).total_seconds() for f in  time_a_050_1])
time_a_050_2 = np.array([(f-time_a_050_2[0]).total_seconds() for f in  time_a_050_2])
time_a_050_3 = np.array([(f-time_a_050_3[0]).total_seconds() for f in  time_a_050_3])

time_max_a_150_1=time_a_150_1[np.nonzero(tau_a_150_1==max(tau_a_150_1))][0]
time_max_a_150_2=time_a_150_2[np.nonzero(tau_a_150_2==max(tau_a_150_2))][0]
time_max_a_150_3=time_a_150_3[np.nonzero(tau_a_150_3==max(tau_a_150_3))][0]
time_max_a_100_1=time_a_100_1[np.nonzero(tau_a_100_1==max(tau_a_100_1))][0]
time_max_a_100_2=time_a_100_2[np.nonzero(tau_a_100_2==max(tau_a_100_2))][0]
time_max_a_100_3=time_a_100_3[np.nonzero(tau_a_100_3==max(tau_a_100_3))][0]
time_max_a_050_1=time_a_050_1[np.nonzero(tau_a_050_1==max(tau_a_050_1))][0]
time_max_a_050_2=time_a_050_2[np.nonzero(tau_a_050_2==max(tau_a_050_2))][0]
time_max_a_050_3=time_a_050_3[np.nonzero(tau_a_050_3==max(tau_a_050_3))][0]

#% Aca  ya salian como np.array asi que es mas facil
time_c_150_1 -= time_c_150_1[0]
time_c_150_2 -= time_c_150_2[0]
time_c_150_3 -= time_c_150_3[0]
time_c_100_1 -= time_c_100_1[0]
time_c_100_2 -= time_c_100_2[0]
time_c_100_3 -= time_c_100_3[0]
time_c_050_1 -= time_c_050_1[0]
time_c_050_2 -= time_c_050_2[0]
time_c_050_3 -= time_c_050_3[0]

time_max_c_150_1=time_c_150_1[np.nonzero(tau_c_150_1==max(tau_c_150_1))][0]
time_max_c_150_2=time_c_150_2[np.nonzero(tau_c_150_2==max(tau_c_150_2))][0]
time_max_c_150_3=time_c_150_3[np.nonzero(tau_c_150_3==max(tau_c_150_3))][0]
time_max_c_100_1=time_c_100_1[np.nonzero(tau_c_100_1==max(tau_c_100_1))][0]
time_max_c_100_2=time_c_100_2[np.nonzero(tau_c_100_2==max(tau_c_100_2))][0]
time_max_c_100_3=time_c_100_3[np.nonzero(tau_c_100_3==max(tau_c_100_3))][0]
time_max_c_050_1=time_c_050_1[np.nonzero(tau_c_050_1==max(tau_c_050_1))][0]
time_max_c_050_2=time_c_050_2[np.nonzero(tau_c_050_2==max(tau_c_050_2))][0]
time_max_c_050_3=time_c_050_3[np.nonzero(tau_c_050_3==max(tau_c_050_3))][0]

#%% Tau vs tiempo
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(time_a_150_1,tau_a_150_1)
ax[0,0].plot(time_a_150_2,tau_a_150_2)
ax[0,0].plot(time_a_150_3,tau_a_150_3)
ax[0,0].scatter(time_max_a_150_1,tau_max_a_150_1,marker='X',label=f'{tau_max_a_150_1:.1f} ns at {time_max_a_150_1:.1f} s')
ax[0,0].scatter(time_max_a_150_2,tau_max_a_150_2,marker='X',label=f'{tau_max_a_150_2:.1f} ns at {time_max_a_150_2:.1f} s')
ax[0,0].scatter(time_max_a_150_3,tau_max_a_150_3,marker='X',label=f'{tau_max_a_150_3:.1f} ns at {time_max_a_150_3:.1f} s')

ax[0,1].plot(time_c_150_1,tau_c_150_1)
ax[0,1].plot(time_c_150_2,tau_c_150_2)
ax[0,1].plot(time_c_150_3,tau_c_150_3)
ax[0,1].scatter(time_max_c_150_1,tau_max_c_150_1,marker='X',label=f'{tau_max_c_150_1:.1f} ns at {time_max_c_150_1:.1f} s')
ax[0,1].scatter(time_max_c_150_2,tau_max_c_150_2,marker='X',label=f'{tau_max_c_150_2:.1f} ns at {time_max_c_150_2:.1f} s')
ax[0,1].scatter(time_max_c_150_3,tau_max_c_150_3,marker='X',label=f'{tau_max_c_150_3:.1f} ns at {time_max_c_150_3:.1f} s')

ax[1,0].plot(time_a_100_1,tau_a_100_1)
ax[1,0].plot(time_a_100_2,tau_a_100_2)
ax[1,0].plot(time_a_100_3,tau_a_100_3)
ax[1,0].scatter(time_max_a_100_1,tau_max_a_100_1,marker='X',label=f'{tau_max_a_100_1:.1f} ns at {time_max_a_100_1:.1f} s')
ax[1,0].scatter(time_max_a_100_2,tau_max_a_100_2,marker='X',label=f'{tau_max_a_100_2:.1f} ns at {time_max_a_100_2:.1f} s')
ax[1,0].scatter(time_max_a_100_3,tau_max_a_100_3,marker='X',label=f'{tau_max_a_100_3:.1f} ns at {time_max_a_100_3:.1f} s')

ax[1,1].plot(time_c_100_1,tau_c_100_1)
ax[1,1].plot(time_c_100_2,tau_c_100_2)
ax[1,1].plot(time_c_100_3,tau_c_100_3)
ax[1,1].scatter(time_max_c_100_1,tau_max_c_100_1,marker='X',label=f'{tau_max_c_100_1:.1f} ns at {time_max_c_100_1:.1f} s')
ax[1,1].scatter(time_max_c_100_2,tau_max_c_100_2,marker='X',label=f'{tau_max_c_100_2:.1f} ns at {time_max_c_100_2:.1f} s')
ax[1,1].scatter(time_max_c_100_3,tau_max_c_100_3,marker='X',label=f'{tau_max_c_100_3:.1f} ns at {time_max_c_100_3:.1f} s')

ax[2,0].plot(time_a_050_1,tau_a_050_1,)
ax[2,0].plot(time_a_050_2,tau_a_050_2,)
ax[2,0].plot(time_a_050_3,tau_a_050_3,)
ax[2,0].scatter(time_max_a_050_1,tau_max_a_050_1,marker='X',label=f'{tau_max_a_050_1:.1f} ns at {time_max_a_050_1:.1f} s')
ax[2,0].scatter(time_max_a_050_2,tau_max_a_050_2,marker='X',label=f'{tau_max_a_050_2:.1f} ns at {time_max_a_050_2:.1f} s')
ax[2,0].scatter(time_max_a_050_3,tau_max_a_050_3,marker='X',label=f'{tau_max_a_050_3:.1f} ns at {time_max_a_050_3:.1f} s')

ax[2,1].plot(time_c_050_1,tau_c_050_1,)
ax[2,1].plot(time_c_050_2,tau_c_050_2,)
ax[2,1].plot(time_c_050_3,tau_c_050_3,)
ax[2,1].scatter(time_max_c_050_1,tau_max_c_050_1,marker='X',label=f'{tau_max_c_050_1:.1f} ns at {time_max_c_050_1:.1f} s')
ax[2,1].scatter(time_max_c_050_2,tau_max_c_050_2,marker='X',label=f'{tau_max_c_050_2:.1f} ns at {time_max_c_050_2:.1f} s')
ax[2,1].scatter(time_max_c_050_3,tau_max_c_050_3,marker='X',label=f'{tau_max_c_050_3:.1f} ns at {time_max_c_050_3:.1f} s')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=9)
        b.set_xlabel('t (s)')
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')   

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_a, ha='center',fontsize=13)
fig.text(0.775, 0.92,dir_c, ha='center',fontsize=13)
plt.suptitle('NE5X cong s/ $H_{DC}^{\perp}$\n$\\tau$ vs tiempo',fontsize=16)

#%% Templogs
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(time_a_150_1,temp_a_150_1)
ax[0,0].plot(time_a_150_2,temp_a_150_2)
ax[0,0].plot(time_a_150_3,temp_a_150_3)
ax[0,0].scatter(time_max_a_150_1,temp_max_a_150_1,marker='X',label=f'{tau_max_a_150_1:.1f} ns at {time_max_a_150_1:.1f} s')
ax[0,0].scatter(time_max_a_150_2,temp_max_a_150_2,marker='X',label=f'{tau_max_a_150_2:.1f} ns at {time_max_a_150_2:.1f} s')
ax[0,0].scatter(time_max_a_150_3,temp_max_a_150_3,marker='X',label=f'{tau_max_a_150_3:.1f} ns at {time_max_a_150_3:.1f} s')

ax[0,1].plot(time_c_150_1,temp_c_150_1)
ax[0,1].plot(time_c_150_2,temp_c_150_2)
ax[0,1].plot(time_c_150_3,temp_c_150_3)
ax[0,1].scatter(time_max_c_150_1,temp_max_c_150_1,marker='X',label=f'{tau_max_c_150_1:.1f} ns at {time_max_c_150_1:.1f} s')
ax[0,1].scatter(time_max_c_150_2,temp_max_c_150_2,marker='X',label=f'{tau_max_c_150_2:.1f} ns at {time_max_c_150_2:.1f} s')
ax[0,1].scatter(time_max_c_150_3,temp_max_c_150_3,marker='X',label=f'{tau_max_c_150_3:.1f} ns at {time_max_c_150_3:.1f} s')

ax[1,0].plot(time_a_100_1,temp_a_100_1)
ax[1,0].plot(time_a_100_2,temp_a_100_2)
ax[1,0].plot(time_a_100_3,temp_a_100_3)
ax[1,0].scatter(time_max_a_100_1,temp_max_a_100_1,marker='X',label=f'{tau_max_a_100_1:.1f} ns at {time_max_a_100_1:.1f} s')
ax[1,0].scatter(time_max_a_100_2,temp_max_a_100_2,marker='X',label=f'{tau_max_a_100_2:.1f} ns at {time_max_a_100_2:.1f} s')
ax[1,0].scatter(time_max_a_100_3,temp_max_a_100_3,marker='X',label=f'{tau_max_a_100_3:.1f} ns at {time_max_a_100_3:.1f} s')

ax[1,1].plot(time_c_100_1,temp_c_100_1)
ax[1,1].plot(time_c_100_2,temp_c_100_2)
ax[1,1].plot(time_c_100_3,temp_c_100_3)
ax[1,1].scatter(time_max_c_100_1,temp_max_c_100_1,marker='X',label=f'{tau_max_c_100_1:.1f} ns at {time_max_c_100_1:.1f} s')
ax[1,1].scatter(time_max_c_100_2,temp_max_c_100_2,marker='X',label=f'{tau_max_c_100_2:.1f} ns at {time_max_c_100_2:.1f} s')
ax[1,1].scatter(time_max_c_100_3,temp_max_c_100_3,marker='X',label=f'{tau_max_c_100_3:.1f} ns at {time_max_c_100_3:.1f} s')

ax[2,0].plot(time_a_050_1,temp_a_050_1,)
ax[2,0].plot(time_a_050_2,temp_a_050_2,)
ax[2,0].plot(time_a_050_3,temp_a_050_3,)
ax[2,0].scatter(time_max_a_050_1,temp_max_a_050_1,marker='X',label=f'{tau_max_a_050_1:.1f} ns at {time_max_a_050_1:.1f} s')
ax[2,0].scatter(time_max_a_050_2,temp_max_a_050_2,marker='X',label=f'{tau_max_a_050_2:.1f} ns at {time_max_a_050_2:.1f} s')
ax[2,0].scatter(time_max_a_050_3,temp_max_a_050_3,marker='X',label=f'{tau_max_a_050_3:.1f} ns at {time_max_a_050_3:.1f} s')

ax[2,1].plot(time_c_050_1,temp_c_050_1,)
ax[2,1].plot(time_c_050_2,temp_c_050_2,)
ax[2,1].plot(time_c_050_3,temp_c_050_3,)
ax[2,1].scatter(time_max_c_050_1,temp_max_c_050_1,marker='X',label=f'{tau_max_c_050_1:.1f} ns at {time_max_c_050_1:.1f} s')
ax[2,1].scatter(time_max_c_050_2,temp_max_c_050_2,marker='X',label=f'{tau_max_c_050_2:.1f} ns at {time_max_c_050_2:.1f} s')
ax[2,1].scatter(time_max_c_050_3,temp_max_c_050_3,marker='X',label=f'{tau_max_c_050_3:.1f} ns at {time_max_c_050_3:.1f} s')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=10)
        b.set_xlabel('t (s)')    
        
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')   

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_a, ha='center',fontsize=13)
fig.text(0.775, 0.92,dir_c, ha='center',fontsize=13)
plt.suptitle('NE5X cong s/ $H_{DC}^{\perp}$\nTemperatura vs tiempo',fontsize=16)

#%% (b) vs (c): 135 kHz  descong sin/con campo DC - distintas fechas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dir_b= 'b - 04-Jul - 135 kHz'
files_b_150 = glob(os.path.join(dir_b, '*150dA*'))
files_b_100 = glob(os.path.join(dir_b, '*100dA*'))
files_b_050 = glob(os.path.join(dir_b, '*050dA*'))

_,files_b_150_1,time_b_150_1,temp_b_150_1,_,_,_,_,_,_,_,_,SAR_b_150_1,tau_b_150_1,_ = lector_resultados(files_b_150[0])
_,files_b_150_2,time_b_150_2,temp_b_150_2,_,_,_,_,_,_,_,_,SAR_b_150_2,tau_b_150_2,_ = lector_resultados(files_b_150[1])
_,files_b_150_3,time_b_150_3,temp_b_150_3,_,_,_,_,_,_,_,_,SAR_b_150_3,tau_b_150_3,_ = lector_resultados(files_b_150[2])

_,files_b_100_1,time_b_100_1,temp_b_100_1,_,_,_,_,_,_,_,_,SAR_b_100_1,tau_b_100_1,_ = lector_resultados(files_b_100[0])
_,files_b_100_2,time_b_100_2,temp_b_100_2,_,_,_,_,_,_,_,_,SAR_b_100_2,tau_b_100_2,_ = lector_resultados(files_b_100[1])
_,files_b_100_3,time_b_100_3,temp_b_100_3,_,_,_,_,_,_,_,_,SAR_b_100_3,tau_b_100_3,_ = lector_resultados(files_b_100[2])

_,files_b_050_1,time_b_050_1,temp_b_050_1,_,_,_,_,_,_,_,_,SAR_b_050_1,tau_b_050_1,_ = lector_resultados(files_b_050[0])
_,files_b_050_2,time_b_050_2,temp_b_050_2,_,_,_,_,_,_,_,_,SAR_b_050_2,tau_b_050_2,_ = lector_resultados(files_b_050[1])
_,files_b_050_3,time_b_050_3,temp_b_050_3,_,_,_,_,_,_,_,_,SAR_b_050_3,tau_b_050_3,_ = lector_resultados(files_b_050[2])

dir_c= 'c - 18-Jul - 135 kHz'
files_c_150 = glob(os.path.join(dir_c, '*150dA*'))
files_c_100 = glob(os.path.join(dir_c, '*100dA*'))
files_c_050 = glob(os.path.join(dir_c, '*050dA*'))

_,files_c_150_1,time_c_150_1,temp_c_150_1,_,_,_,_,_,_,_,_,SAR_c_150_1,tau_c_150_1,_ = lector_resultados_2(files_c_150[0])
_,files_c_150_2,time_c_150_2,temp_c_150_2,_,_,_,_,_,_,_,_,SAR_c_150_2,tau_c_150_2,_ = lector_resultados_2(files_c_150[1])
_,files_c_150_3,time_c_150_3,temp_c_150_3,_,_,_,_,_,_,_,_,SAR_c_150_3,tau_c_150_3,_ = lector_resultados_2(files_c_150[2])

_,files_c_100_1,time_c_100_1,temp_c_100_1,_,_,_,_,_,_,_,_,SAR_c_100_1,tau_c_100_1,_ = lector_resultados_2(files_c_100[0])
_,files_c_100_2,time_c_100_2,temp_c_100_2,_,_,_,_,_,_,_,_,SAR_c_100_2,tau_c_100_2,_ = lector_resultados_2(files_c_100[1])
_,files_c_100_3,time_c_100_3,temp_c_100_3,_,_,_,_,_,_,_,_,SAR_c_100_3,tau_c_100_3,_ = lector_resultados_2(files_c_100[2])

_,files_c_050_1,time_c_050_1,temp_c_050_1,_,_,_,_,_,_,_,_,SAR_c_050_1,tau_c_050_1,_ = lector_resultados_2(files_c_050[0])
_,files_c_050_2,time_c_050_2,temp_c_050_2,_,_,_,_,_,_,_,_,SAR_c_050_2,tau_c_050_2,_ = lector_resultados_2(files_c_050[1])
_,files_c_050_3,time_c_050_3,temp_c_050_3,_,_,_,_,_,_,_,_,SAR_c_050_3,tau_c_050_3,_ = lector_resultados_2(files_c_050[2])

#% paso tau a ns
tau_b_150_1=tau_b_150_1*1e9
tau_b_150_2=tau_b_150_2*1e9
tau_b_150_3=tau_b_150_3*1e9
tau_b_100_1=tau_b_100_1*1e9
tau_b_100_2=tau_b_100_2*1e9
tau_b_100_3=tau_b_100_3*1e9
tau_b_050_1=tau_b_050_1*1e9
tau_b_050_2=tau_b_050_2*1e9
tau_b_050_3=tau_b_050_3*1e9

tau_c_150_1=tau_c_150_1*1e9
tau_c_150_2=tau_c_150_2*1e9
tau_c_150_3=tau_c_150_3*1e9
tau_c_100_1=tau_c_100_1*1e9
tau_c_100_2=tau_c_100_2*1e9
tau_c_100_3=tau_c_100_3*1e9
tau_c_050_1=tau_c_050_1*1e9
tau_c_050_2=tau_c_050_2*1e9
tau_c_050_3=tau_c_050_3*1e9
#%encuentro los maximos 
tau_max_b_150_1=tau_b_150_1[np.nonzero(tau_b_150_1==max(tau_b_150_1))][0]
tau_max_b_150_2=tau_b_150_2[np.nonzero(tau_b_150_2==max(tau_b_150_2))][0]
tau_max_b_150_3=tau_b_150_3[np.nonzero(tau_b_150_3==max(tau_b_150_3))][0]
tau_max_b_100_1=tau_b_100_1[np.nonzero(tau_b_100_1==max(tau_b_100_1))][0]
tau_max_b_100_2=tau_b_100_2[np.nonzero(tau_b_100_2==max(tau_b_100_2))][0]
tau_max_b_100_3=tau_b_100_3[np.nonzero(tau_b_100_3==max(tau_b_100_3))][0]
tau_max_b_050_1=tau_b_050_1[np.nonzero(tau_b_050_1==max(tau_b_050_1))][0]
tau_max_b_050_2=tau_b_050_2[np.nonzero(tau_b_050_2==max(tau_b_050_2))][0]
tau_max_b_050_3=tau_b_050_3[np.nonzero(tau_b_050_3==max(tau_b_050_3))][0]

temp_max_b_150_1=temp_b_150_1[np.nonzero(tau_b_150_1==max(tau_b_150_1))][0]
temp_max_b_150_2=temp_b_150_2[np.nonzero(tau_b_150_2==max(tau_b_150_2))][0]
temp_max_b_150_3=temp_b_150_3[np.nonzero(tau_b_150_3==max(tau_b_150_3))][0]
temp_max_b_100_1=temp_b_100_1[np.nonzero(tau_b_100_1==max(tau_b_100_1))][0]
temp_max_b_100_2=temp_b_100_2[np.nonzero(tau_b_100_2==max(tau_b_100_2))][0]
temp_max_b_100_3=temp_b_100_3[np.nonzero(tau_b_100_3==max(tau_b_100_3))][0]
temp_max_b_050_1=temp_b_050_1[np.nonzero(tau_b_050_1==max(tau_b_050_1))][0]
temp_max_b_050_2=temp_b_050_2[np.nonzero(tau_b_050_2==max(tau_b_050_2))][0]
temp_max_b_050_3=temp_b_050_3[np.nonzero(tau_b_050_3==max(tau_b_050_3))][0]

tau_max_c_150_1=tau_c_150_1[np.nonzero(tau_c_150_1==max(tau_c_150_1))][0]
tau_max_c_150_2=tau_c_150_2[np.nonzero(tau_c_150_2==max(tau_c_150_2))][0]
tau_max_c_150_3=tau_c_150_3[np.nonzero(tau_c_150_3==max(tau_c_150_3))][0]
tau_max_c_100_1=tau_c_100_1[np.nonzero(tau_c_100_1==max(tau_c_100_1))][0]
tau_max_c_100_2=tau_c_100_2[np.nonzero(tau_c_100_2==max(tau_c_100_2))][0]
tau_max_c_100_3=tau_c_100_3[np.nonzero(tau_c_100_3==max(tau_c_100_3))][0]
tau_max_c_050_1=tau_c_050_1[np.nonzero(tau_c_050_1==max(tau_c_050_1))][0]
tau_max_c_050_2=tau_c_050_2[np.nonzero(tau_c_050_2==max(tau_c_050_2))][0]
tau_max_c_050_3=tau_c_050_3[np.nonzero(tau_c_050_3==max(tau_c_050_3))][0]

temp_max_c_150_1=temp_c_150_1[np.nonzero(tau_c_150_1==max(tau_c_150_1))][0]
temp_max_c_150_2=temp_c_150_2[np.nonzero(tau_c_150_2==max(tau_c_150_2))][0]
temp_max_c_150_3=temp_c_150_3[np.nonzero(tau_c_150_3==max(tau_c_150_3))][0]
temp_max_c_100_1=temp_c_100_1[np.nonzero(tau_c_100_1==max(tau_c_100_1))][0]
temp_max_c_100_2=temp_c_100_2[np.nonzero(tau_c_100_2==max(tau_c_100_2))][0]
temp_max_c_100_3=temp_c_100_3[np.nonzero(tau_c_100_3==max(tau_c_100_3))][0]
temp_max_c_050_1=temp_c_050_1[np.nonzero(tau_c_050_1==max(tau_c_050_1))][0]
temp_max_c_050_2=temp_c_050_2[np.nonzero(tau_c_050_2==max(tau_c_050_2))][0]
temp_max_c_050_3=temp_c_050_3[np.nonzero(tau_c_050_3==max(tau_c_050_3))][0]

#%% Tau vs T 
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(temp_b_150_1,tau_b_150_1)
ax[0,0].plot(temp_b_150_2,tau_b_150_2)
ax[0,0].plot(temp_b_150_3,tau_b_150_3)
ax[0,0].scatter(temp_max_b_150_1,tau_max_b_150_1,marker='v',label=f'{tau_max_b_150_1:.1f} ns at {temp_max_b_150_1}°C')
ax[0,0].scatter(temp_max_b_150_2,tau_max_b_150_2,marker='v',label=f'{tau_max_b_150_2:.1f} ns at {temp_max_b_150_2}°C')
ax[0,0].scatter(temp_max_b_150_3,tau_max_b_150_3,marker='v',label=f'{tau_max_b_150_3:.1f} ns at {temp_max_b_150_3}°C')

ax[0,1].plot(temp_c_150_1,tau_c_150_1)
ax[0,1].plot(temp_c_150_2,tau_c_150_2)
ax[0,1].plot(temp_c_150_3,tau_c_150_3)
ax[0,1].scatter(temp_max_c_150_1,tau_max_c_150_1,marker='v',label=f'{tau_max_c_150_1:.1f} ns at {temp_max_c_150_1}°C')
ax[0,1].scatter(temp_max_c_150_2,tau_max_c_150_2,marker='v',label=f'{tau_max_c_150_2:.1f} ns at {temp_max_c_150_2}°C')
ax[0,1].scatter(temp_max_c_150_3,tau_max_c_150_3,marker='v',label=f'{tau_max_c_150_3:.1f} ns at {temp_max_c_150_3}°C')

ax[1,0].plot(temp_b_100_1,tau_b_100_1)
ax[1,0].plot(temp_b_100_2,tau_b_100_2)
ax[1,0].plot(temp_b_100_3,tau_b_100_3)
ax[1,0].scatter(temp_max_b_100_1,tau_max_b_100_1,marker='v',label=f'{tau_max_b_100_1:.1f} ns at {temp_max_b_100_1}°C')
ax[1,0].scatter(temp_max_b_100_2,tau_max_b_100_2,marker='v',label=f'{tau_max_b_100_2:.1f} ns at {temp_max_b_100_2}°C')
ax[1,0].scatter(temp_max_b_100_3,tau_max_b_100_3,marker='v',label=f'{tau_max_b_100_3:.1f} ns at {temp_max_b_100_3}°C')

ax[1,1].plot(temp_c_100_1,tau_c_100_1)
ax[1,1].plot(temp_c_100_2,tau_c_100_2)
ax[1,1].plot(temp_c_100_3,tau_c_100_3)
ax[1,1].scatter(temp_max_c_100_1,tau_max_c_100_1,marker='v',label=f'{tau_max_c_100_1:.1f} ns at {temp_max_c_100_1}°C')
ax[1,1].scatter(temp_max_c_100_2,tau_max_c_100_2,marker='v',label=f'{tau_max_c_100_2:.1f} ns at {temp_max_c_100_2}°C')
ax[1,1].scatter(temp_max_c_100_3,tau_max_c_100_3,marker='v',label=f'{tau_max_c_100_3:.1f} ns at {temp_max_c_100_3}°C')

ax[2,0].plot(temp_b_050_1,tau_b_050_1,)
ax[2,0].plot(temp_b_050_2,tau_b_050_2,)
ax[2,0].plot(temp_b_050_3,tau_b_050_3,)
ax[2,0].scatter(temp_max_b_050_1,tau_max_b_050_1,marker='v',label=f'{tau_max_b_050_1:.1f} ns at {temp_max_b_050_1}°C')
ax[2,0].scatter(temp_max_b_050_2,tau_max_b_050_2,marker='v',label=f'{tau_max_b_050_2:.1f} ns at {temp_max_b_050_2}°C')
ax[2,0].scatter(temp_max_b_050_3,tau_max_b_050_3,marker='v',label=f'{tau_max_b_050_3:.1f} ns at {temp_max_b_050_3}°C')

ax[2,1].plot(temp_c_050_1,tau_c_050_1,)
ax[2,1].plot(temp_c_050_2,tau_c_050_2,)
ax[2,1].plot(temp_c_050_3,tau_c_050_3,)
ax[2,1].scatter(temp_max_c_050_1,tau_max_c_050_1,marker='v',label=f'{tau_max_c_050_1:.1f} ns at {temp_max_c_050_1}°C')
ax[2,1].scatter(temp_max_c_050_2,tau_max_c_050_2,marker='v',label=f'{tau_max_c_050_2:.1f} ns at {temp_max_c_050_2}°C')
ax[2,1].scatter(temp_max_c_050_3,tau_max_c_050_3,marker='v',label=f'{tau_max_c_050_3:.1f} ns at {temp_max_c_050_3}°C')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=9)
        b.set_xlabel('T (°C)')
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_b+' - c/ $H_{DC}^{\perp}$', ha='center',fontsize=12)
fig.text(0.775, 0.92,dir_c+' - s/ $H_{DC}^{\perp}$', ha='center',fontsize=12)
plt.suptitle('NE5X cong s/c $H_{DC}$\n$\\tau$ vs Temperatura',fontsize=16)
#% Tiempos absoultos
time_b_150_1 = np.array([(f-time_b_150_1[0]).total_seconds() for f in  time_b_150_1])
time_b_150_2 = np.array([(f-time_b_150_2[0]).total_seconds() for f in  time_b_150_2])
time_b_150_3 = np.array([(f-time_b_150_3[0]).total_seconds() for f in  time_b_150_3])
time_b_100_1 = np.array([(f-time_b_100_1[0]).total_seconds() for f in  time_b_100_1])
time_b_100_2 = np.array([(f-time_b_100_2[0]).total_seconds() for f in  time_b_100_2])
time_b_100_3 = np.array([(f-time_b_100_3[0]).total_seconds() for f in  time_b_100_3])
time_b_050_1 = np.array([(f-time_b_050_1[0]).total_seconds() for f in  time_b_050_1])
time_b_050_2 = np.array([(f-time_b_050_2[0]).total_seconds() for f in  time_b_050_2])
time_b_050_3 = np.array([(f-time_b_050_3[0]).total_seconds() for f in  time_b_050_3])

time_max_b_150_1=time_b_150_1[np.nonzero(tau_b_150_1==max(tau_b_150_1))][0]
time_max_b_150_2=time_b_150_2[np.nonzero(tau_b_150_2==max(tau_b_150_2))][0]
time_max_b_150_3=time_b_150_3[np.nonzero(tau_b_150_3==max(tau_b_150_3))][0]
time_max_b_100_1=time_b_100_1[np.nonzero(tau_b_100_1==max(tau_b_100_1))][0]
time_max_b_100_2=time_b_100_2[np.nonzero(tau_b_100_2==max(tau_b_100_2))][0]
time_max_b_100_3=time_b_100_3[np.nonzero(tau_b_100_3==max(tau_b_100_3))][0]
time_max_b_050_1=time_b_050_1[np.nonzero(tau_b_050_1==max(tau_b_050_1))][0]
time_max_b_050_2=time_b_050_2[np.nonzero(tau_b_050_2==max(tau_b_050_2))][0]
time_max_b_050_3=time_b_050_3[np.nonzero(tau_b_050_3==max(tau_b_050_3))][0]

#% Aca  ya salian como np.array asi que es mas facil
time_c_150_1 -= time_c_150_1[0]
time_c_150_2 -= time_c_150_2[0]
time_c_150_3 -= time_c_150_3[0]
time_c_100_1 -= time_c_100_1[0]
time_c_100_2 -= time_c_100_2[0]
time_c_100_3 -= time_c_100_3[0]
time_c_050_1 -= time_c_050_1[0]
time_c_050_2 -= time_c_050_2[0]
time_c_050_3 -= time_c_050_3[0]

time_max_c_150_1=time_c_150_1[np.nonzero(tau_c_150_1==max(tau_c_150_1))][0]
time_max_c_150_2=time_c_150_2[np.nonzero(tau_c_150_2==max(tau_c_150_2))][0]
time_max_c_150_3=time_c_150_3[np.nonzero(tau_c_150_3==max(tau_c_150_3))][0]
time_max_c_100_1=time_c_100_1[np.nonzero(tau_c_100_1==max(tau_c_100_1))][0]
time_max_c_100_2=time_c_100_2[np.nonzero(tau_c_100_2==max(tau_c_100_2))][0]
time_max_c_100_3=time_c_100_3[np.nonzero(tau_c_100_3==max(tau_c_100_3))][0]
time_max_c_050_1=time_c_050_1[np.nonzero(tau_c_050_1==max(tau_c_050_1))][0]
time_max_c_050_2=time_c_050_2[np.nonzero(tau_c_050_2==max(tau_c_050_2))][0]
time_max_c_050_3=time_c_050_3[np.nonzero(tau_c_050_3==max(tau_c_050_3))][0]

#%% Tau vs tiempo
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(time_b_150_1,tau_b_150_1)
ax[0,0].plot(time_b_150_2,tau_b_150_2)
ax[0,0].plot(time_b_150_3,tau_b_150_3)
ax[0,0].scatter(time_max_b_150_1,tau_max_b_150_1,marker='X',label=f'{tau_max_b_150_1:.1f} ns at {time_max_b_150_1:.1f} s')
ax[0,0].scatter(time_max_b_150_2,tau_max_b_150_2,marker='X',label=f'{tau_max_b_150_2:.1f} ns at {time_max_b_150_2:.1f} s')
ax[0,0].scatter(time_max_b_150_3,tau_max_b_150_3,marker='X',label=f'{tau_max_b_150_3:.1f} ns at {time_max_b_150_3:.1f} s')

ax[0,1].plot(time_c_150_1,tau_c_150_1)
ax[0,1].plot(time_c_150_2,tau_c_150_2)
ax[0,1].plot(time_c_150_3,tau_c_150_3)
ax[0,1].scatter(time_max_c_150_1,tau_max_c_150_1,marker='X',label=f'{tau_max_c_150_1:.1f} ns at {time_max_c_150_1:.1f} s')
ax[0,1].scatter(time_max_c_150_2,tau_max_c_150_2,marker='X',label=f'{tau_max_c_150_2:.1f} ns at {time_max_c_150_2:.1f} s')
ax[0,1].scatter(time_max_c_150_3,tau_max_c_150_3,marker='X',label=f'{tau_max_c_150_3:.1f} ns at {time_max_c_150_3:.1f} s')

ax[1,0].plot(time_b_100_1,tau_b_100_1)
ax[1,0].plot(time_b_100_2,tau_b_100_2)
ax[1,0].plot(time_b_100_3,tau_b_100_3)
ax[1,0].scatter(time_max_b_100_1,tau_max_b_100_1,marker='X',label=f'{tau_max_b_100_1:.1f} ns at {time_max_b_100_1:.1f} s')
ax[1,0].scatter(time_max_b_100_2,tau_max_b_100_2,marker='X',label=f'{tau_max_b_100_2:.1f} ns at {time_max_b_100_2:.1f} s')
ax[1,0].scatter(time_max_b_100_3,tau_max_b_100_3,marker='X',label=f'{tau_max_b_100_3:.1f} ns at {time_max_b_100_3:.1f} s')

ax[1,1].plot(time_c_100_1,tau_c_100_1)
ax[1,1].plot(time_c_100_2,tau_c_100_2)
ax[1,1].plot(time_c_100_3,tau_c_100_3)
ax[1,1].scatter(time_max_c_100_1,tau_max_c_100_1,marker='X',label=f'{tau_max_c_100_1:.1f} ns at {time_max_c_100_1:.1f} s')
ax[1,1].scatter(time_max_c_100_2,tau_max_c_100_2,marker='X',label=f'{tau_max_c_100_2:.1f} ns at {time_max_c_100_2:.1f} s')
ax[1,1].scatter(time_max_c_100_3,tau_max_c_100_3,marker='X',label=f'{tau_max_c_100_3:.1f} ns at {time_max_c_100_3:.1f} s')

ax[2,0].plot(time_b_050_1,tau_b_050_1,)
ax[2,0].plot(time_b_050_2,tau_b_050_2,)
ax[2,0].plot(time_b_050_3,tau_b_050_3,)
ax[2,0].scatter(time_max_b_050_1,tau_max_b_050_1,marker='X',label=f'{tau_max_b_050_1:.1f} ns at {time_max_b_050_1:.1f} s')
ax[2,0].scatter(time_max_b_050_2,tau_max_b_050_2,marker='X',label=f'{tau_max_b_050_2:.1f} ns at {time_max_b_050_2:.1f} s')
ax[2,0].scatter(time_max_b_050_3,tau_max_b_050_3,marker='X',label=f'{tau_max_b_050_3:.1f} ns at {time_max_b_050_3:.1f} s')

ax[2,1].plot(time_c_050_1,tau_c_050_1,)
ax[2,1].plot(time_c_050_2,tau_c_050_2,)
ax[2,1].plot(time_c_050_3,tau_c_050_3,)
ax[2,1].scatter(time_max_c_050_1,tau_max_c_050_1,marker='X',label=f'{tau_max_c_050_1:.1f} ns at {time_max_c_050_1:.1f} s')
ax[2,1].scatter(time_max_c_050_2,tau_max_c_050_2,marker='X',label=f'{tau_max_c_050_2:.1f} ns at {time_max_c_050_2:.1f} s')
ax[2,1].scatter(time_max_c_050_3,tau_max_c_050_3,marker='X',label=f'{tau_max_c_050_3:.1f} ns at {time_max_c_050_3:.1f} s')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=9)
        b.set_xlabel('t (s)')
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')   

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_b+' - c/ $H_{DC}^{\perp}$', ha='center',fontsize=12)
fig.text(0.775, 0.92,dir_c+' - s/ $H_{DC}^{\perp}$', ha='center',fontsize=12)
plt.suptitle('NE5X cong s/c $H_{DC}$\n $\\tau$ vs tiempo',fontsize=16)

#%% Templogs
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(time_b_150_1,temp_b_150_1)
ax[0,0].plot(time_b_150_2,temp_b_150_2)
ax[0,0].plot(time_b_150_3,temp_b_150_3)
ax[0,0].scatter(time_max_b_150_1,temp_max_b_150_1,marker='X',label=f'{tau_max_b_150_1:.1f} ns at {time_max_b_150_1:.1f} s')
ax[0,0].scatter(time_max_b_150_2,temp_max_b_150_2,marker='X',label=f'{tau_max_b_150_2:.1f} ns at {time_max_b_150_2:.1f} s')
ax[0,0].scatter(time_max_b_150_3,temp_max_b_150_3,marker='X',label=f'{tau_max_b_150_3:.1f} ns at {time_max_b_150_3:.1f} s')

ax[0,1].plot(time_c_150_1,temp_c_150_1)
ax[0,1].plot(time_c_150_2,temp_c_150_2)
ax[0,1].plot(time_c_150_3,temp_c_150_3)
ax[0,1].scatter(time_max_c_150_1,temp_max_c_150_1,marker='X',label=f'{tau_max_c_150_1:.1f} ns at {time_max_c_150_1:.1f} s')
ax[0,1].scatter(time_max_c_150_2,temp_max_c_150_2,marker='X',label=f'{tau_max_c_150_2:.1f} ns at {time_max_c_150_2:.1f} s')
ax[0,1].scatter(time_max_c_150_3,temp_max_c_150_3,marker='X',label=f'{tau_max_c_150_3:.1f} ns at {time_max_c_150_3:.1f} s')

ax[1,0].plot(time_b_100_1,temp_b_100_1)
ax[1,0].plot(time_b_100_2,temp_b_100_2)
ax[1,0].plot(time_b_100_3,temp_b_100_3)
ax[1,0].scatter(time_max_b_100_1,temp_max_b_100_1,marker='X',label=f'{tau_max_b_100_1:.1f} ns at {time_max_b_100_1:.1f} s')
ax[1,0].scatter(time_max_b_100_2,temp_max_b_100_2,marker='X',label=f'{tau_max_b_100_2:.1f} ns at {time_max_b_100_2:.1f} s')
ax[1,0].scatter(time_max_b_100_3,temp_max_b_100_3,marker='X',label=f'{tau_max_b_100_3:.1f} ns at {time_max_b_100_3:.1f} s')

ax[1,1].plot(time_c_100_1,temp_c_100_1)
ax[1,1].plot(time_c_100_2,temp_c_100_2)
ax[1,1].plot(time_c_100_3,temp_c_100_3)
ax[1,1].scatter(time_max_c_100_1,temp_max_c_100_1,marker='X',label=f'{tau_max_c_100_1:.1f} ns at {time_max_c_100_1:.1f} s')
ax[1,1].scatter(time_max_c_100_2,temp_max_c_100_2,marker='X',label=f'{tau_max_c_100_2:.1f} ns at {time_max_c_100_2:.1f} s')
ax[1,1].scatter(time_max_c_100_3,temp_max_c_100_3,marker='X',label=f'{tau_max_c_100_3:.1f} ns at {time_max_c_100_3:.1f} s')

ax[2,0].plot(time_b_050_1,temp_b_050_1,)
ax[2,0].plot(time_b_050_2,temp_b_050_2,)
ax[2,0].plot(time_b_050_3,temp_b_050_3,)
ax[2,0].scatter(time_max_b_050_1,temp_max_b_050_1,marker='X',label=f'{tau_max_b_050_1:.1f} ns at {time_max_b_050_1:.1f} s')
ax[2,0].scatter(time_max_b_050_2,temp_max_b_050_2,marker='X',label=f'{tau_max_b_050_2:.1f} ns at {time_max_b_050_2:.1f} s')
ax[2,0].scatter(time_max_b_050_3,temp_max_b_050_3,marker='X',label=f'{tau_max_b_050_3:.1f} ns at {time_max_b_050_3:.1f} s')

ax[2,1].plot(time_c_050_1,temp_c_050_1,)
ax[2,1].plot(time_c_050_2,temp_c_050_2,)
ax[2,1].plot(time_c_050_3,temp_c_050_3,)
ax[2,1].scatter(time_max_c_050_1,temp_max_c_050_1,marker='X',label=f'{tau_max_c_050_1:.1f} ns at {time_max_c_050_1:.1f} s')
ax[2,1].scatter(time_max_c_050_2,temp_max_c_050_2,marker='X',label=f'{tau_max_c_050_2:.1f} ns at {time_max_c_050_2:.1f} s')
ax[2,1].scatter(time_max_c_050_3,temp_max_c_050_3,marker='X',label=f'{tau_max_c_050_3:.1f} ns at {time_max_c_050_3:.1f} s')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=10)
        b.set_xlabel('t (s)')    
        
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')   

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_b+' - c/ $H_{DC}^{\perp}$', ha='center',fontsize=12)
fig.text(0.775, 0.92,dir_c+' - s/ $H_{DC}^{\perp}$', ha='center',fontsize=12)
plt.suptitle('NE5X cong s/c $H_{DC}$\nTemperatura vs tiempo',fontsize=16)


#%% (a) vs (b): 265 kHz sin/con campo DC - distintas fechas  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_a= 'a  - 26-Jun - 265 kHz'
files_a_150 = glob(os.path.join(dir_a, '*150dA*'))
files_a_100 = glob(os.path.join(dir_a, '*100dA*'))
files_a_050 = glob(os.path.join(dir_a, '*050dA*'))

_,files_a_150_1,time_a_150_1,temp_a_150_1,_,_,_,_,_,_,_,_,SAR_a_150_1,tau_a_150_1,_ = lector_resultados(files_a_150[0])
_,files_a_150_2,time_a_150_2,temp_a_150_2,_,_,_,_,_,_,_,_,SAR_a_150_2,tau_a_150_2,_ = lector_resultados(files_a_150[1])
_,files_a_150_3,time_a_150_3,temp_a_150_3,_,_,_,_,_,_,_,_,SAR_a_150_3,tau_a_150_3,_ = lector_resultados(files_a_150[2])

_,files_a_100_1,time_a_100_1,temp_a_100_1,_,_,_,_,_,_,_,_,SAR_a_100_1,tau_a_100_1,_ = lector_resultados(files_a_100[0])
_,files_a_100_2,time_a_100_2,temp_a_100_2,_,_,_,_,_,_,_,_,SAR_a_100_2,tau_a_100_2,_ = lector_resultados(files_a_100[1])
_,files_a_100_3,time_a_100_3,temp_a_100_3,_,_,_,_,_,_,_,_,SAR_a_100_3,tau_a_100_3,_ = lector_resultados(files_a_100[2])

_,files_a_050_1,time_a_050_1,temp_a_050_1,_,_,_,_,_,_,_,_,SAR_a_050_1,tau_a_050_1,_ = lector_resultados(files_a_050[0])
_,files_a_050_2,time_a_050_2,temp_a_050_2,_,_,_,_,_,_,_,_,SAR_a_050_2,tau_a_050_2,_ = lector_resultados(files_a_050[1])
_,files_a_050_3,time_a_050_3,temp_a_050_3,_,_,_,_,_,_,_,_,SAR_a_050_3,tau_a_050_3,_ = lector_resultados(files_a_050[2])

dir_b= 'b - 04-Jul - 265 kHz'
files_b_150 = glob(os.path.join(dir_b, '*150dA*'))
files_b_100 = glob(os.path.join(dir_b, '*100dA*'))
files_b_050 = glob(os.path.join(dir_b, '*050dA*'))

_,files_b_150_1,time_b_150_1,temp_b_150_1,_,_,_,_,_,_,_,_,SAR_b_150_1,tau_b_150_1,_ = lector_resultados(files_b_150[0])
_,files_b_150_2,time_b_150_2,temp_b_150_2,_,_,_,_,_,_,_,_,SAR_b_150_2,tau_b_150_2,_ = lector_resultados(files_b_150[1])
_,files_b_150_3,time_b_150_3,temp_b_150_3,_,_,_,_,_,_,_,_,SAR_b_150_3,tau_b_150_3,_ = lector_resultados(files_b_150[2])

_,files_b_100_1,time_b_100_1,temp_b_100_1,_,_,_,_,_,_,_,_,SAR_b_100_1,tau_b_100_1,_ = lector_resultados(files_b_100[0])
_,files_b_100_2,time_b_100_2,temp_b_100_2,_,_,_,_,_,_,_,_,SAR_b_100_2,tau_b_100_2,_ = lector_resultados(files_b_100[1])
_,files_b_100_3,time_b_100_3,temp_b_100_3,_,_,_,_,_,_,_,_,SAR_b_100_3,tau_b_100_3,_ = lector_resultados(files_b_100[2])

_,files_b_050_1,time_b_050_1,temp_b_050_1,_,_,_,_,_,_,_,_,SAR_b_050_1,tau_b_050_1,_ = lector_resultados(files_b_050[0])
_,files_b_050_2,time_b_050_2,temp_b_050_2,_,_,_,_,_,_,_,_,SAR_b_050_2,tau_b_050_2,_ = lector_resultados(files_b_050[1])
_,files_b_050_3,time_b_050_3,temp_b_050_3,_,_,_,_,_,_,_,_,SAR_b_050_3,tau_b_050_3,_ = lector_resultados(files_b_050[2])

#% paso tau a ns
tau_a_150_1=tau_a_150_1*1e9
tau_a_150_2=tau_a_150_2*1e9
tau_a_150_3=tau_a_150_3*1e9
tau_a_100_1=tau_a_100_1*1e9
tau_a_100_2=tau_a_100_2*1e9
tau_a_100_3=tau_a_100_3*1e9
tau_a_050_1=tau_a_050_1*1e9
tau_a_050_2=tau_a_050_2*1e9
tau_a_050_3=tau_a_050_3*1e9

tau_b_150_1=tau_b_150_1*1e9
tau_b_150_2=tau_b_150_2*1e9
tau_b_150_3=tau_b_150_3*1e9
tau_b_100_1=tau_b_100_1*1e9
tau_b_100_2=tau_b_100_2*1e9
tau_b_100_3=tau_b_100_3*1e9
tau_b_050_1=tau_b_050_1*1e9
tau_b_050_2=tau_b_050_2*1e9
tau_b_050_3=tau_b_050_3*1e9
#%encuentro los maximos 
tau_max_a_150_1=tau_a_150_1[np.nonzero(tau_a_150_1==max(tau_a_150_1))][0]
tau_max_a_150_2=tau_a_150_2[np.nonzero(tau_a_150_2==max(tau_a_150_2))][0]
tau_max_a_150_3=tau_a_150_3[np.nonzero(tau_a_150_3==max(tau_a_150_3))][0]
tau_max_a_100_1=tau_a_100_1[np.nonzero(tau_a_100_1==max(tau_a_100_1))][0]
tau_max_a_100_2=tau_a_100_2[np.nonzero(tau_a_100_2==max(tau_a_100_2))][0]
tau_max_a_100_3=tau_a_100_3[np.nonzero(tau_a_100_3==max(tau_a_100_3))][0]
tau_max_a_050_1=tau_a_050_1[np.nonzero(tau_a_050_1==max(tau_a_050_1))][0]
tau_max_a_050_2=tau_a_050_2[np.nonzero(tau_a_050_2==max(tau_a_050_2))][0]
tau_max_a_050_3=tau_a_050_3[np.nonzero(tau_a_050_3==max(tau_a_050_3))][0]

temp_max_a_150_1=temp_a_150_1[np.nonzero(tau_a_150_1==max(tau_a_150_1))][0]
temp_max_a_150_2=temp_a_150_2[np.nonzero(tau_a_150_2==max(tau_a_150_2))][0]
temp_max_a_150_3=temp_a_150_3[np.nonzero(tau_a_150_3==max(tau_a_150_3))][0]
temp_max_a_100_1=temp_a_100_1[np.nonzero(tau_a_100_1==max(tau_a_100_1))][0]
temp_max_a_100_2=temp_a_100_2[np.nonzero(tau_a_100_2==max(tau_a_100_2))][0]
temp_max_a_100_3=temp_a_100_3[np.nonzero(tau_a_100_3==max(tau_a_100_3))][0]
temp_max_a_050_1=temp_a_050_1[np.nonzero(tau_a_050_1==max(tau_a_050_1))][0]
temp_max_a_050_2=temp_a_050_2[np.nonzero(tau_a_050_2==max(tau_a_050_2))][0]
temp_max_a_050_3=temp_a_050_3[np.nonzero(tau_a_050_3==max(tau_a_050_3))][0]

tau_max_b_150_1=tau_b_150_1[np.nonzero(tau_b_150_1==max(tau_b_150_1))][0]
tau_max_b_150_2=tau_b_150_2[np.nonzero(tau_b_150_2==max(tau_b_150_2))][0]
tau_max_b_150_3=tau_b_150_3[np.nonzero(tau_b_150_3==max(tau_b_150_3))][0]
tau_max_b_100_1=tau_b_100_1[np.nonzero(tau_b_100_1==max(tau_b_100_1))][0]
tau_max_b_100_2=tau_b_100_2[np.nonzero(tau_b_100_2==max(tau_b_100_2))][0]
tau_max_b_100_3=tau_b_100_3[np.nonzero(tau_b_100_3==max(tau_b_100_3))][0]
tau_max_b_050_1=tau_b_050_1[np.nonzero(tau_b_050_1==max(tau_b_050_1))][0]
tau_max_b_050_2=tau_b_050_2[np.nonzero(tau_b_050_2==max(tau_b_050_2))][0]
tau_max_b_050_3=tau_b_050_3[np.nonzero(tau_b_050_3==max(tau_b_050_3))][0]

temp_max_b_150_1=temp_b_150_1[np.nonzero(tau_b_150_1==max(tau_b_150_1))][0]
temp_max_b_150_2=temp_b_150_2[np.nonzero(tau_b_150_2==max(tau_b_150_2))][0]
temp_max_b_150_3=temp_b_150_3[np.nonzero(tau_b_150_3==max(tau_b_150_3))][0]
temp_max_b_100_1=temp_b_100_1[np.nonzero(tau_b_100_1==max(tau_b_100_1))][0]
temp_max_b_100_2=temp_b_100_2[np.nonzero(tau_b_100_2==max(tau_b_100_2))][0]
temp_max_b_100_3=temp_b_100_3[np.nonzero(tau_b_100_3==max(tau_b_100_3))][0]
temp_max_b_050_1=temp_b_050_1[np.nonzero(tau_b_050_1==max(tau_b_050_1))][0]
temp_max_b_050_2=temp_b_050_2[np.nonzero(tau_b_050_2==max(tau_b_050_2))][0]
temp_max_b_050_3=temp_b_050_3[np.nonzero(tau_b_050_3==max(tau_b_050_3))][0]

#%% Tau vs T
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(temp_a_150_1,tau_a_150_1)
ax[0,0].plot(temp_a_150_2,tau_a_150_2)
ax[0,0].plot(temp_a_150_3,tau_a_150_3)
ax[0,0].scatter(temp_max_a_150_1,tau_max_a_150_1,marker='v',label=f'{tau_max_a_150_1:.1f} ns at {temp_max_a_150_1}°C')
ax[0,0].scatter(temp_max_a_150_2,tau_max_a_150_2,marker='v',label=f'{tau_max_a_150_2:.1f} ns at {temp_max_a_150_2}°C')
ax[0,0].scatter(temp_max_a_150_3,tau_max_a_150_3,marker='v',label=f'{tau_max_a_150_3:.1f} ns at {temp_max_a_150_3}°C')

ax[0,1].plot(temp_b_150_1,tau_b_150_1)
ax[0,1].plot(temp_b_150_2,tau_b_150_2)
ax[0,1].plot(temp_b_150_3,tau_b_150_3)
ax[0,1].scatter(temp_max_b_150_1,tau_max_b_150_1,marker='v',label=f'{tau_max_b_150_1:.1f} ns at {temp_max_b_150_1}°C')
ax[0,1].scatter(temp_max_b_150_2,tau_max_b_150_2,marker='v',label=f'{tau_max_b_150_2:.1f} ns at {temp_max_b_150_2}°C')
ax[0,1].scatter(temp_max_b_150_3,tau_max_b_150_3,marker='v',label=f'{tau_max_b_150_3:.1f} ns at {temp_max_b_150_3}°C')

ax[1,0].plot(temp_a_100_1,tau_a_100_1)
ax[1,0].plot(temp_a_100_2,tau_a_100_2)
ax[1,0].plot(temp_a_100_3,tau_a_100_3)
ax[1,0].scatter(temp_max_a_100_1,tau_max_a_100_1,marker='v',label=f'{tau_max_a_100_1:.1f} ns at {temp_max_a_100_1}°C')
ax[1,0].scatter(temp_max_a_100_2,tau_max_a_100_2,marker='v',label=f'{tau_max_a_100_2:.1f} ns at {temp_max_a_100_2}°C')
ax[1,0].scatter(temp_max_a_100_3,tau_max_a_100_3,marker='v',label=f'{tau_max_a_100_3:.1f} ns at {temp_max_a_100_3}°C')

ax[1,1].plot(temp_b_100_1,tau_b_100_1)
ax[1,1].plot(temp_b_100_2,tau_b_100_2)
ax[1,1].plot(temp_b_100_3,tau_b_100_3)
ax[1,1].scatter(temp_max_b_100_1,tau_max_b_100_1,marker='v',label=f'{tau_max_b_100_1:.1f} ns at {temp_max_b_100_1}°C')
ax[1,1].scatter(temp_max_b_100_2,tau_max_b_100_2,marker='v',label=f'{tau_max_b_100_2:.1f} ns at {temp_max_b_100_2}°C')
ax[1,1].scatter(temp_max_b_100_3,tau_max_b_100_3,marker='v',label=f'{tau_max_b_100_3:.1f} ns at {temp_max_b_100_3}°C')

ax[2,0].plot(temp_a_050_1,tau_a_050_1,)
ax[2,0].plot(temp_a_050_2,tau_a_050_2,)
ax[2,0].plot(temp_a_050_3,tau_a_050_3,)
ax[2,0].scatter(temp_max_a_050_1,tau_max_a_050_1,marker='v',label=f'{tau_max_a_050_1:.1f} ns at {temp_max_a_050_1}°C')
ax[2,0].scatter(temp_max_a_050_2,tau_max_a_050_2,marker='v',label=f'{tau_max_a_050_2:.1f} ns at {temp_max_a_050_2}°C')
ax[2,0].scatter(temp_max_a_050_3,tau_max_a_050_3,marker='v',label=f'{tau_max_a_050_3:.1f} ns at {temp_max_a_050_3}°C')

ax[2,1].plot(temp_b_050_1,tau_b_050_1,)
ax[2,1].plot(temp_b_050_2,tau_b_050_2,)
ax[2,1].plot(temp_b_050_3,tau_b_050_3,)
ax[2,1].scatter(temp_max_b_050_1,tau_max_b_050_1,marker='v',label=f'{tau_max_b_050_1:.1f} ns at {temp_max_b_050_1}°C')
ax[2,1].scatter(temp_max_b_050_2,tau_max_b_050_2,marker='v',label=f'{tau_max_b_050_2:.1f} ns at {temp_max_b_050_2}°C')
ax[2,1].scatter(temp_max_b_050_3,tau_max_b_050_3,marker='v',label=f'{tau_max_b_050_3:.1f} ns at {temp_max_b_050_3}°C')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=9)
        b.set_xlabel('T (°C)')
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_a+' - s/ $H_{DC}^{\perp}$', ha='center',fontsize=13)
fig.text(0.775, 0.92,dir_b+' - c/ $H_{DC}^{\perp}$', ha='center',fontsize=13)
plt.suptitle('NE5X cong s/ $H_{DC}^{\perp}$\n$\\tau$ vs Temperatura',fontsize=16)

#%% Tiempos absoultos
time_a_150_1 = np.array([(f-time_a_150_1[0]).total_seconds() for f in  time_a_150_1])
time_a_150_2 = np.array([(f-time_a_150_2[0]).total_seconds() for f in  time_a_150_2])
time_a_150_3 = np.array([(f-time_a_150_3[0]).total_seconds() for f in  time_a_150_3])
time_a_100_1 = np.array([(f-time_a_100_1[0]).total_seconds() for f in  time_a_100_1])
time_a_100_2 = np.array([(f-time_a_100_2[0]).total_seconds() for f in  time_a_100_2])
time_a_100_3 = np.array([(f-time_a_100_3[0]).total_seconds() for f in  time_a_100_3])
time_a_050_1 = np.array([(f-time_a_050_1[0]).total_seconds() for f in  time_a_050_1])
time_a_050_2 = np.array([(f-time_a_050_2[0]).total_seconds() for f in  time_a_050_2])
time_a_050_3 = np.array([(f-time_a_050_3[0]).total_seconds() for f in  time_a_050_3])

time_max_a_150_1=time_a_150_1[np.nonzero(tau_a_150_1==max(tau_a_150_1))][0]
time_max_a_150_2=time_a_150_2[np.nonzero(tau_a_150_2==max(tau_a_150_2))][0]
time_max_a_150_3=time_a_150_3[np.nonzero(tau_a_150_3==max(tau_a_150_3))][0]
time_max_a_100_1=time_a_100_1[np.nonzero(tau_a_100_1==max(tau_a_100_1))][0]
time_max_a_100_2=time_a_100_2[np.nonzero(tau_a_100_2==max(tau_a_100_2))][0]
time_max_a_100_3=time_a_100_3[np.nonzero(tau_a_100_3==max(tau_a_100_3))][0]
time_max_a_050_1=time_a_050_1[np.nonzero(tau_a_050_1==max(tau_a_050_1))][0]
time_max_a_050_2=time_a_050_2[np.nonzero(tau_a_050_2==max(tau_a_050_2))][0]
time_max_a_050_3=time_a_050_3[np.nonzero(tau_a_050_3==max(tau_a_050_3))][0]

time_b_150_1 = np.array([(f-time_b_150_1[0]).total_seconds() for f in  time_b_150_1])
time_b_150_2 = np.array([(f-time_b_150_2[0]).total_seconds() for f in  time_b_150_2])
time_b_150_3 = np.array([(f-time_b_150_3[0]).total_seconds() for f in  time_b_150_3])
time_b_100_1 = np.array([(f-time_b_100_1[0]).total_seconds() for f in  time_b_100_1])
time_b_100_2 = np.array([(f-time_b_100_2[0]).total_seconds() for f in  time_b_100_2])
time_b_100_3 = np.array([(f-time_b_100_3[0]).total_seconds() for f in  time_b_100_3])
time_b_050_1 = np.array([(f-time_b_050_1[0]).total_seconds() for f in  time_b_050_1])
time_b_050_2 = np.array([(f-time_b_050_2[0]).total_seconds() for f in  time_b_050_2])
time_b_050_3 = np.array([(f-time_b_050_3[0]).total_seconds() for f in  time_b_050_3])

time_max_b_150_1=time_b_150_1[np.nonzero(tau_b_150_1==max(tau_b_150_1))][0]
time_max_b_150_2=time_b_150_2[np.nonzero(tau_b_150_2==max(tau_b_150_2))][0]
time_max_b_150_3=time_b_150_3[np.nonzero(tau_b_150_3==max(tau_b_150_3))][0]
time_max_b_100_1=time_b_100_1[np.nonzero(tau_b_100_1==max(tau_b_100_1))][0]
time_max_b_100_2=time_b_100_2[np.nonzero(tau_b_100_2==max(tau_b_100_2))][0]
time_max_b_100_3=time_b_100_3[np.nonzero(tau_b_100_3==max(tau_b_100_3))][0]
time_max_b_050_1=time_b_050_1[np.nonzero(tau_b_050_1==max(tau_b_050_1))][0]
time_max_b_050_2=time_b_050_2[np.nonzero(tau_b_050_2==max(tau_b_050_2))][0]
time_max_b_050_3=time_b_050_3[np.nonzero(tau_b_050_3==max(tau_b_050_3))][0]

#%% Tau vs tiempo
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(time_a_150_1,tau_a_150_1)
ax[0,0].plot(time_a_150_2,tau_a_150_2)
ax[0,0].plot(time_a_150_3,tau_a_150_3)
ax[0,0].scatter(time_max_a_150_1,tau_max_a_150_1,marker='X',label=f'{tau_max_a_150_1:.1f} ns at {time_max_a_150_1:.1f} s')
ax[0,0].scatter(time_max_a_150_2,tau_max_a_150_2,marker='X',label=f'{tau_max_a_150_2:.1f} ns at {time_max_a_150_2:.1f} s')
ax[0,0].scatter(time_max_a_150_3,tau_max_a_150_3,marker='X',label=f'{tau_max_a_150_3:.1f} ns at {time_max_a_150_3:.1f} s')

ax[0,1].plot(time_b_150_1,tau_b_150_1)
ax[0,1].plot(time_b_150_2,tau_b_150_2)
ax[0,1].plot(time_b_150_3,tau_b_150_3)
ax[0,1].scatter(time_max_b_150_1,tau_max_b_150_1,marker='X',label=f'{tau_max_b_150_1:.1f} ns at {time_max_b_150_1:.1f} s')
ax[0,1].scatter(time_max_b_150_2,tau_max_b_150_2,marker='X',label=f'{tau_max_b_150_2:.1f} ns at {time_max_b_150_2:.1f} s')
ax[0,1].scatter(time_max_b_150_3,tau_max_b_150_3,marker='X',label=f'{tau_max_b_150_3:.1f} ns at {time_max_b_150_3:.1f} s')

ax[1,0].plot(time_a_100_1,tau_a_100_1)
ax[1,0].plot(time_a_100_2,tau_a_100_2)
ax[1,0].plot(time_a_100_3,tau_a_100_3)
ax[1,0].scatter(time_max_a_100_1,tau_max_a_100_1,marker='X',label=f'{tau_max_a_100_1:.1f} ns at {time_max_a_100_1:.1f} s')
ax[1,0].scatter(time_max_a_100_2,tau_max_a_100_2,marker='X',label=f'{tau_max_a_100_2:.1f} ns at {time_max_a_100_2:.1f} s')
ax[1,0].scatter(time_max_a_100_3,tau_max_a_100_3,marker='X',label=f'{tau_max_a_100_3:.1f} ns at {time_max_a_100_3:.1f} s')

ax[1,1].plot(time_b_100_1,tau_b_100_1)
ax[1,1].plot(time_b_100_2,tau_b_100_2)
ax[1,1].plot(time_b_100_3,tau_b_100_3)
ax[1,1].scatter(time_max_b_100_1,tau_max_b_100_1,marker='X',label=f'{tau_max_b_100_1:.1f} ns at {time_max_b_100_1:.1f} s')
ax[1,1].scatter(time_max_b_100_2,tau_max_b_100_2,marker='X',label=f'{tau_max_b_100_2:.1f} ns at {time_max_b_100_2:.1f} s')
ax[1,1].scatter(time_max_b_100_3,tau_max_b_100_3,marker='X',label=f'{tau_max_b_100_3:.1f} ns at {time_max_b_100_3:.1f} s')

ax[2,0].plot(time_a_050_1,tau_a_050_1,)
ax[2,0].plot(time_a_050_2,tau_a_050_2,)
ax[2,0].plot(time_a_050_3,tau_a_050_3,)
ax[2,0].scatter(time_max_a_050_1,tau_max_a_050_1,marker='X',label=f'{tau_max_a_050_1:.1f} ns at {time_max_a_050_1:.1f} s')
ax[2,0].scatter(time_max_a_050_2,tau_max_a_050_2,marker='X',label=f'{tau_max_a_050_2:.1f} ns at {time_max_a_050_2:.1f} s')
ax[2,0].scatter(time_max_a_050_3,tau_max_a_050_3,marker='X',label=f'{tau_max_a_050_3:.1f} ns at {time_max_a_050_3:.1f} s')

ax[2,1].plot(time_b_050_1,tau_b_050_1,)
ax[2,1].plot(time_b_050_2,tau_b_050_2,)
ax[2,1].plot(time_b_050_3,tau_b_050_3,)
ax[2,1].scatter(time_max_b_050_1,tau_max_b_050_1,marker='X',label=f'{tau_max_b_050_1:.1f} ns at {time_max_b_050_1:.1f} s')
ax[2,1].scatter(time_max_b_050_2,tau_max_b_050_2,marker='X',label=f'{tau_max_b_050_2:.1f} ns at {time_max_b_050_2:.1f} s')
ax[2,1].scatter(time_max_b_050_3,tau_max_b_050_3,marker='X',label=f'{tau_max_b_050_3:.1f} ns at {time_max_b_050_3:.1f} s')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=9)
        b.set_xlabel('t (s)')
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')   

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_a+' - s/ $H_{DC}^{\perp}$', ha='center',fontsize=13)
fig.text(0.775, 0.92,dir_b+' - c/ $H_{DC}^{\perp}$', ha='center',fontsize=13)
plt.suptitle('NE5X cong s/ $H_{DC}^{\perp}$\n$\\tau$ vs tiempo',fontsize=16)

#%% Templogs
fig, ax = plt.subplots(ncols=2,nrows=3,figsize=(10,9),constrained_layout=True,sharey='row',sharex='row')
ax[0,0].plot(time_a_150_1,temp_a_150_1)
ax[0,0].plot(time_a_150_2,temp_a_150_2)
ax[0,0].plot(time_a_150_3,temp_a_150_3)
ax[0,0].scatter(time_max_a_150_1,temp_max_a_150_1,marker='X',label=f'{tau_max_a_150_1:.1f} ns at {time_max_a_150_1:.1f} s')
ax[0,0].scatter(time_max_a_150_2,temp_max_a_150_2,marker='X',label=f'{tau_max_a_150_2:.1f} ns at {time_max_a_150_2:.1f} s')
ax[0,0].scatter(time_max_a_150_3,temp_max_a_150_3,marker='X',label=f'{tau_max_a_150_3:.1f} ns at {time_max_a_150_3:.1f} s')

ax[0,1].plot(time_b_150_1,temp_b_150_1)
ax[0,1].plot(time_b_150_2,temp_b_150_2)
ax[0,1].plot(time_b_150_3,temp_b_150_3)
ax[0,1].scatter(time_max_b_150_1,temp_max_b_150_1,marker='X',label=f'{tau_max_b_150_1:.1f} ns at {time_max_b_150_1:.1f} s')
ax[0,1].scatter(time_max_b_150_2,temp_max_b_150_2,marker='X',label=f'{tau_max_b_150_2:.1f} ns at {time_max_b_150_2:.1f} s')
ax[0,1].scatter(time_max_b_150_3,temp_max_b_150_3,marker='X',label=f'{tau_max_b_150_3:.1f} ns at {time_max_b_150_3:.1f} s')

ax[1,0].plot(time_a_100_1,temp_a_100_1)
ax[1,0].plot(time_a_100_2,temp_a_100_2)
ax[1,0].plot(time_a_100_3,temp_a_100_3)
ax[1,0].scatter(time_max_a_100_1,temp_max_a_100_1,marker='X',label=f'{tau_max_a_100_1:.1f} ns at {time_max_a_100_1:.1f} s')
ax[1,0].scatter(time_max_a_100_2,temp_max_a_100_2,marker='X',label=f'{tau_max_a_100_2:.1f} ns at {time_max_a_100_2:.1f} s')
ax[1,0].scatter(time_max_a_100_3,temp_max_a_100_3,marker='X',label=f'{tau_max_a_100_3:.1f} ns at {time_max_a_100_3:.1f} s')

ax[1,1].plot(time_b_100_1,temp_b_100_1)
ax[1,1].plot(time_b_100_2,temp_b_100_2)
ax[1,1].plot(time_b_100_3,temp_b_100_3)
ax[1,1].scatter(time_max_b_100_1,temp_max_b_100_1,marker='X',label=f'{tau_max_b_100_1:.1f} ns at {time_max_b_100_1:.1f} s')
ax[1,1].scatter(time_max_b_100_2,temp_max_b_100_2,marker='X',label=f'{tau_max_b_100_2:.1f} ns at {time_max_b_100_2:.1f} s')
ax[1,1].scatter(time_max_b_100_3,temp_max_b_100_3,marker='X',label=f'{tau_max_b_100_3:.1f} ns at {time_max_b_100_3:.1f} s')

ax[2,0].plot(time_a_050_1,temp_a_050_1,)
ax[2,0].plot(time_a_050_2,temp_a_050_2,)
ax[2,0].plot(time_a_050_3,temp_a_050_3,)
ax[2,0].scatter(time_max_a_050_1,temp_max_a_050_1,marker='X',label=f'{tau_max_a_050_1:.1f} ns at {time_max_a_050_1:.1f} s')
ax[2,0].scatter(time_max_a_050_2,temp_max_a_050_2,marker='X',label=f'{tau_max_a_050_2:.1f} ns at {time_max_a_050_2:.1f} s')
ax[2,0].scatter(time_max_a_050_3,temp_max_a_050_3,marker='X',label=f'{tau_max_a_050_3:.1f} ns at {time_max_a_050_3:.1f} s')

ax[2,1].plot(time_b_050_1,temp_b_050_1,)
ax[2,1].plot(time_b_050_2,temp_b_050_2,)
ax[2,1].plot(time_b_050_3,temp_b_050_3,)
ax[2,1].scatter(time_max_b_050_1,temp_max_b_050_1,marker='X',label=f'{tau_max_b_050_1:.1f} ns at {time_max_b_050_1:.1f} s')
ax[2,1].scatter(time_max_b_050_2,temp_max_b_050_2,marker='X',label=f'{tau_max_b_050_2:.1f} ns at {time_max_b_050_2:.1f} s')
ax[2,1].scatter(time_max_b_050_3,temp_max_b_050_3,marker='X',label=f'{tau_max_b_050_3:.1f} ns at {time_max_b_050_3:.1f} s')

for a in ax:
    for b in a:
        b.grid()
        b.legend(title=r'$\tau_{max}$',fontsize=10)
        b.set_xlabel('t (s)')    
        
for a in ax:
    a[0].set_ylabel(r'$\tau$ (ns)')   

ax[0,0].set_title('57 kA/m',loc='left') 
ax[0,1].set_title('57 kA/m',loc='left')
ax[1,0].set_title('38 kA/m',loc='left') 
ax[1,1].set_title('38 kA/m',loc='left')
ax[2,0].set_title('20 kA/m',loc='left') 
ax[2,1].set_title('20 kA/m',loc='left')

fig.text(0.275, 0.92,dir_a+' - s/ $H_{DC}^{\perp}$', ha='center',fontsize=13)
fig.text(0.775, 0.92,dir_b+' - c/ $H_{DC}^{\perp}$', ha='center',fontsize=13)
plt.suptitle('NE5X cong s/ $H_{DC}$\nTemperatura vs tiempo',fontsize=16)


