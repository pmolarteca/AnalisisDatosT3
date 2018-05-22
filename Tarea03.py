# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:03:40 2017

@author: palom
"""

#tarea 3 curso An'alisis de datos



from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.basemap 
from mpl_toolkits.basemap import Basemap
import datetime
import matplotlib.colors

from scipy import stats
import scipy.stats as scp


Pathout='C:/Users/palom/Documents/AnalisisDatos/Tarea03/'
Data=Dataset('precip.mon.mean.nc','r')
DataSST=Dataset('sst.mnmean.nc','r')
mask=Dataset('lsmask.nc','r')


print Data.variables
print DataSST.variables
print Data.variables.keys()
print DataSST.variables['sst']
print Data.variables['precip']
print DataSST.variables['time']
print mask.variables['mask']


precip = np.array (Data.variables['precip'][:])
precip[precip==-9.96921e+36]=np.nan
lat = np.array (Data.variables['lat'][:])
lon = np.array (Data.variables['lon'][:])
time = np.array (Data.variables['time'][:])
time = time.astype(float)




sst = np.array (DataSST.variables['sst'][:])
sst[sst==32767]=np.nan
lat_sst = np.array (DataSST.variables['lat'][:])
lon_sst = np.array (DataSST.variables['lon'][:])
time_sst = np.array (DataSST.variables['time'][:])

maska=np.array(mask.variables['mask'])

sstmask=maska*sst


fechas = []
for i in range(len(time)):
#print i
#print time[i]
    fechas.append(datetime.datetime(1800,01,01)+datetime.timedelta(days = time[i]))

fechas = np.array(fechas)
print fechas

#------------------------------------------------------------

fechas_sst = []
for i in range(len(time_sst)):
#print i
#print time[i]
    fechas_sst.append(datetime.datetime(1800,01,01)+datetime.timedelta(days = time_sst[i]))

fechas_sst = np.array(fechas_sst)
print fechas_sst




#cambiar resolucion

latsst_new=np.flipud(lat_sst)
sst_new = np.flipud(sst_new)


lats_fine = np.arange(latsst_new[0], latsst_new[-1], 2.5) 
lons_fine = np.arange(lon_sst[0], lon_sst[-1], 2.5)
lons_sub, lats_sub = np.meshgrid(lons_fine, lats_fine)


tempe=np.zeros((430,72,144))*np.NaN

for i in range (len(time_sst)): 
    mapasst=sstmask[[i][0],:,:]
    
    sst_nuevo=mpl_toolkits.basemap.interp(mapasst, lon_sst, latsst_new, lons_sub, lats_sub, order=1)
    tempe[[i][0],:,:]=sst_nuevo
   

#recorto los datos para dejarlo de la misma longitud y en la zona 40N-40S

P_fechaini=np.where(fechas == datetime.datetime(1981,12,1))[0][0]
T_fechafin=np.where(fechas_sst== datetime.datetime(2017,7,1))[0][0]

precipit=precip[P_fechaini:]
temper=tempe[:T_fechafin+1]

Lat40 = np.where((lats_fine > -40) & (lats_fine < 40))[0]
Lat40P=np.where((lat> -40) & (lat < 40))[0]
lats_40=lats_fine[Lat40]
lats_40P=lat[Lat40P]

sst_40=temper[:,Lat40,:]
ppt_40=precipit[:,Lat40P,:]

#comparo matriz nueva con datos originales 

a = np.where(fechas_sst == datetime.datetime(2015,6,1))   
   
mapa_dia1_original=sstmask[[0][0],:,:]
mapa_dia1_nuevo=sst_40[[0][0],:,:]
   
fig=plt.figure(figsize=(30,20))
plt.imshow(mapa_dia1_original)

fig=plt.figure(figsize=(30,20))
plt.imshow(mapa_dia1_nuevo)

#graficos

fig=plt.figure(figsize=(30,10))
ax = fig.add_subplot(111)
# Basemap es el paquete que dibuja las líneas del mapa
m = Basemap(llcrnrlat=np.min(lats_40),urcrnrlat=np.max(lats_40), \
llcrnrlon=np.min(lons_fine),urcrnrlon=np.max(lons_fine),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lats_40.shape[0]; nx = lons_fine.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
cs = m.contourf(x,y,np.flipud(CicloAnual[0,:,:]),cmap='seismic')
m.colorbar(location='bottom',pad="10%")
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()



fig=plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)
# Basemap es el paquete que dibuja las líneas del mapa
m = Basemap(llcrnrlat=np.min(lat_sst),urcrnrlat=np.max(lat_sst), \
llcrnrlon=np.min(lon_sst),urcrnrlon=np.max(lon_sst),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lat_sst.shape[0]; nx = lon_sst.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
cs = m.contourf(x,y,np.flipud(mapa_dia1_original),cmap='jet')
m.colorbar(location='bottom',pad="10%")
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()




#correlacion punto a punto


#estandarizar los datos 


fechas_new=fechas_sst[:-2]
 
Meses = np.array([fechas_new[i].month for i in range(len(fechas_new))])
Mapa_Media = np.zeros([12,len(lats_40),len(lons_fine)]) * np.NaN
Mapa_Desv=np.zeros([12,len(lats_40),len(lons_fine)]) * np.NaN
Mapa_Media_Precip=np.zeros([12,len(lats_40),len(lons_fine)]) * np.NaN
Mapa_Desv_Precip=np.zeros([12,len(lats_40),len(lons_fine)]) * np.NaN


for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
    SST= sst_40[tmpp]
    PPT= ppt_40[tmpp]
    for i in range(len(lats_40)):
        for j in range(len(lons_fine)):
            Mapa_Media[k-1,i,j] = np.mean(SST[:,i,j])
            Mapa_Desv[k-1,i,j] = np.std(SST[:,  i,j])
            Mapa_Media_Precip[k-1,i,j]=np.mean(PPT[:,i,j])
            Mapa_Desv_Precip[k-1,i,j]=np.std(PPT[:,i,j])
    
Datos_Est_sst=np.zeros([428,32,144]) *np.NaN
Datos_Est_ppt=np.zeros([428,32,144]) *np.NaN


for i in range(len(sst_40)):
        m=Meses[i]-1
        Datos_Est_sst[i,:,:] = (sst_40[i] - Mapa_Media[m])/Mapa_Desv[m]
        Datos_Est_ppt[i,:,:] = (ppt_40[i] - Mapa_Media_Precip[m])/Mapa_Desv_Precip[m]
   
    
  #  qq3 = ~(np.isnan(EPTriM[:,i])|np.isnan(EUqTriM[:,i]))



    

Mapa_Correlacion = np.zeros([len(lats_40),len(lons_fine)]) * np.NaN

for i in range(len(lats_40)):
    for j in range(len(lons_fine)):
       Mapa_Correlacion[i,j] = scp.pearsonr(Datos_Est_sst[:,i,j], Datos_Est_ppt[:,i,j])[0]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlat=np.min(lats_40),urcrnrlat=np.max(lats_40), \
llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lats_40.shape[0]; nx = lon.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
#cs = m.contourf(x,y,np.flipud(Mapa))
cs = m.contourf(x,y,np.flipud(Datos_Est_sst[0]), cmap='seismic',clevs=20, vmin=-1, vmax=1)
#m.colorbar(location='bottom',pad="10%")
m.colorbar(boundaries=np.linspace(-1,1,5))
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()



Mapa_Cor_Mes=np.zeros([12,len(lats_40),len(lons_fine)]) * np.NaN

for k in range(1,13):
    tmpp = np.where(Meses == k)[0]
    DSST= Datos_Est_sst[tmpp]
    DPPT= Datos_Est_ppt[tmpp]
    for i in range(len(lats_40)):
        for j in range(len(lons_fine)):
            Mapa_Cor_Mes[k-1,i,j] = scp.pearsonr(DSST[:,i,j], DPPT[:,i,j])[0]
      
      
  

namemes=('Ene','Feb','Mar','Abr','May','Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic' )

fig = plt.figure(figsize=(17,28))

for i in range(12):
        
    ax = fig.add_subplot(12,2,i+1)
    ax.set_title(namemes[i])
    m = Basemap(llcrnrlat=np.min(lats_40),urcrnrlat=np.max(lats_40), \
    llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
    rsphere=6371200.,resolution='l',area_thresh=10000)
    ny = lats_40.shape[0]; nx = lon.shape[0]
    lons, lats = m.makegrid(nx, ny)
    x,y = m(lons, lats)
    #cs = m.contourf(x,y,np.flipud(Mapa))
    cs = m.contourf(x,y,np.flipud(Mapa_Cor_Mes[i]), cmap='seismic',clevs=10, vmin=-1, vmax=1)
    #m.colorbar(location='bottom',pad="10%")
    m.colorbar(boundaries=np.linspace(-1,1,5))
    m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
    linewidth=0.1)
    m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
    linewidth=0.1)
    m.drawcoastlines()
    m.drawmapboundary()
    cbar_ax = fig.add_axes([0.96, 0.1, 0.04, 0.8])
    cbar = plt.colorbar(Mapa_Cor_Mes[0],cax=cbar_ax,orientation='vertical')
    cbar.set_label(r'Temperature  $[^\circ C]$',fontsize=14)
    
    


#explorar si es estacionario
#rezagos
#Precipitacion rezagada respecto a la temperatura

Corr_R = np.zeros([12,12,len(lats_40),len(lons_fine)]) * np.NaN
# [Meses, Rezagos, lat,lon]

for Mes in range(1,13):
    tmpo = np.where(Meses == Mes)[0]
    tmpo = tmpo[:-1]
    Temp_Mes = Datos_Est_sst[tmpo,:,:]
    for Rezago in range(0,12):
        Temp_Rezago = Datos_Est_ppt[tmpo + Rezago,:,:]
        for i in range(len(lats_40)):
            for j in range(len(lons_fine)):
                Corr_R[Mes - 1,Rezago,i,j] = \
                scp.pearsonr(Temp_Rezago[:,i,j], Temp_Mes[:,i,j])[0]


#matriz de correlacion

Corr_R_prom=np.nanmean(Corr_R, axis=3)
Corr_R_prom=np.nanmean(Corr_R_prom, axis=2)


#genero la matriz de correlacion
Matriz = np.zeros([12,12]) 

for i in range(0,12):
    for j in range(0,12):
        if j<i:
            Matriz[i,j]=Corr_R_prom[i,j+(12-i)]
        else:   
            Matriz[i,j]=Corr_R_prom[i,j-i]

#graficar matriz de corr


Mes_Ref = 0
fig = plt.figure(figsize=(17,28))
for Rezago in range(0,12):
    ax = fig.add_subplot(12, 2, Rezago+1)
    ax.set_title(u'SST ' +str(namemes[Mes_Ref]) \
    + u' - Precipitación ' + str(namemes[Rezago]))
    # Basemap es el paquete que dibuja las líneas del mapa
    m = Basemap(llcrnrlat=np.min(lats_40),urcrnrlat=np.max(lats_40), \
    llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
    rsphere=6371200.,resolution='l',area_thresh=10000, lon_0 =179)
    ny = lats_40.shape[0]; nx = lons_fine.shape[0]
    lons, lats = m.makegrid(nx, ny)
    x,y = m(lons, lats)
    #cs = m.contourf(x,y,np.flipud(Mapa))
    cs = m.contourf(x,y,np.flipud(Corr_R[Mes_Ref,Rezago,:,:]),cmap='seismic',clevs=10, vmin=-1, vmax=1)
    m.colorbar()
    m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11, linewidth=0.1)
    m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
    linewidth=0.1)
    m.drawcoastlines()
    m.drawmapboundary() 




 
#rezago temperatura repecto a precip


Corr_RR = np.zeros([12,12,len(lats_40),len(lons_fine)]) * np.NaN
# [Meses, Rezagos, lat,lon]

for Mes in range(1,13):
    tmpo = np.where(Meses == Mes)[0]
    tmpo = tmpo[:-1]
    Temp_Mes2 = Datos_Est_ppt[tmpo,:,:]
    print Temp_Mes2
    for Rezago in range(0,12):
        Temp_Rezago2 = Datos_Est_sst[tmpo + Rezago,:,:]
        print Temp_Rezago2
        for i in range(len(lats_40)):
            for j in range(len(lons_fine)):
                Corr_R[Mes - 1,Rezago,i,j] = \
                scp.pearsonr( Temp_Rezago2[:,i,j], Temp_Mes2[:,i,j])[0]




Mes_Ref = 2

for Rezago in range(0,13):
    fig = plt.figure(figsize=(30,12))
    ax = fig.add_subplot(111)
    ax.set_title(u'Correlación Mes = ' +str(Mes_Ref + 1) \
    + ' - Rezago =' + str(Rezago))
    # Basemap es el paquete que dibuja las líneas del mapa
    m = Basemap(llcrnrlat=np.min(lats_40),urcrnrlat=np.max(lats_40), \
    llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
    rsphere=6371200.,resolution='l',area_thresh=10000, lon_0 =179)
    ny = lats_40.shape[0]; nx = lons_fine.shape[0]
    lons, lats = m.makegrid(nx, ny)
    x,y = m(lons, lats)
    #cs = m.contourf(x,y,np.flipud(Mapa))
    cs = m.contourf(x,y,np.flipud(Corr_RR[Mes_Ref,Rezago,:,:]),cmap='bwr',clevs=20, vmin=-1, vmax=1)
    m.colorbar()
    m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11, linewidth=0.1)
    m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
    linewidth=0.1)
    m.drawcoastlines()
    m.drawmapboundary() 







#-------------------------------------------------------------------------------
#seleccino un region de interes
#recorto la matriz de precipitacion


Lat_P = np.where((lat > 5) & (lat < 15))[0]
Lon_P = np.where((lon > 320) & (lon < 340))[0]


ZonaPre=[]

for i in range(len(precip)):

    ZonaP= precip[i,Lat_P,:]
    ZonaP = ZonaP[:,Lon_P]
    ZonaP= ZonaP[np.isfinite(ZonaP)]
    ZonaPre.append(np.mean(ZonaP))


Mapa_Pre_Temp = np.zeros([len(lat),len(lon)]) * np.NaN

for i in range(len(lat)):
    for j in range(len(lon)):
     Mapa_Pre_Temp[i,j] = scp.pearsonr(ZonaPre, temper[:,i,j])[0]


fig = plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lat.shape[0]; nx = lon.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
#cs = m.contourf(x,y,np.flipud(Mapa))
cs = m.contourf(x,y,np.roll(Mapa_Pre_Temp, len(lon)/2))
#m.colorbar(location='bottom',pad="10%")
m.colorbar()
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()




#seleccino un region de interes
#recorto la matriz de precipitacion


Lat_S = np.where((lats_fine > 5) & (lats_fine < 15))[0]
Lon_S = np.where((lons_fine > 320) & (lons_fine < 340))[0]


ZonaTem=[]

for i in range(len(temper)):

    ZonaT= temper[i,Lat_S,:]
    ZonaT = ZonaT[:,Lon_S]
    ZonaT= ZonaT[np.isfinite(ZonaT)]
    ZonaTem.append(np.mean(ZonaT))

ZonaTem=np.array(ZonaTem)

Mapa_Temp_Pre = np.zeros([len(lat),len(lon)]) * np.NaN

for i in range(len(lat)):
    for j in range(len(lon)):
     Mapa_Temp_Pre[i,j] = scp.pearsonr(ZonaTem, precip[:,i,j])[0]



fig = plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlat=np.min(lat),urcrnrlat=np.max(lat), \
llcrnrlon=np.min(0),urcrnrlon=np.max(360),\
rsphere=6371200.,resolution='l',area_thresh=10000)
ny = lat.shape[0]; nx = lon.shape[0]
lons, lats = m.makegrid(nx, ny)
x,y = m(lons, lats)
#cs = m.contourf(x,y,np.flipud(Mapa))
cs = m.contourf(x,y,np.roll(Mapa_Temp_Pre, len(lon)/2))
#m.colorbar(location='bottom',pad="10%")
m.colorbar()
m.drawparallels(np.arange(-90.,90,30.), labels=[1,0,0,0], size=11,\
linewidth=0.1)
m.drawmeridians(np.arange(0, 360, 30.),labels=[0,1,0,1], size=11, \
linewidth=0.1)
m.drawcoastlines()
m.drawmapboundary()


        
        
        
        
        
        

