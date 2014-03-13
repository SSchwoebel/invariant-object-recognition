import nest as n
import nest.raster_plot as rp
import nest.topology as tp
import pylab as pl
import numpy as np

n.ResetKernel()
C_E = 0.5
C_I = 0.2
g_E = 25.
g_I = 20.
V_th = -50.
V_reset = -55.
tau_AMPA = 2.4
tau_NMDA = 100.
tau_GABA = 7.
tau_m_E = C_E/g_E
tau_m_I = C_I/g_I
tau_pre = 20.
tau_post = 10.
c_IE = 0.1
c_EI = 0.024
c_II = 0.1
c_IN = 1.
S_E1E1 = 50.0
sigma_E1E1 = 5.
S_E2E1 = 1.0
sigma_E2E1 = 4.
N_E0 = 30
N_E1 = 30
N_E2 = 10
N_I = 10
N_IN = 1
w_I1E1 = 100.
w_I1I1 = -10.
w_E1I1 = -100.
w_GABA = -10.
w_AMPANMDA = 1.0
w_IN = 0.9
simtime = 30.
xsteps = 20
ysteps = 20

def Write_Data(data,filename):
	with file(filename, 'w') as outfile:
		for i in range(data.shape[0]):
			np.savetxt(outfile, data[i])			
	new_data = np.loadtxt(filename)
	new_data = new_data.reshape((data.shape[0],data.shape[1],data.shape[2]))
	assert np.all(new_data == data)
	
def Plot_Image(img):
	pl.figure()
	pl.imshow(img,cmap='gray')
	pl.show()

# neuron model with slow excitatory synapses for layer E0 and E2
n.CopyModel('iaf_psc_alpha', 'iaf_exc_slow', {'C_m':C_E, 'V_reset':V_reset, 'V_th':V_th, 'tau_m':tau_m_E, 'tau_syn_ex':tau_NMDA, 'tau_syn_in':tau_GABA})
# neuron model with fast excitatory synapses for layer E1
n.CopyModel('iaf_psc_alpha', 'iaf_exc_fast', {'C_m':C_E, 'V_reset':V_reset, 'V_th':V_th, 'tau_m':tau_m_E, 'tau_syn_ex':tau_AMPA, 'tau_syn_in':tau_GABA})
# neuron model with slow excitatory synapses for layer I1
n.CopyModel('iaf_psc_alpha', 'iaf_inh', {'C_m':C_I, 'V_reset':V_reset, 'V_th':V_th, 'tau_m':tau_m_I, 'tau_syn_ex':tau_NMDA, 'tau_syn_in':tau_GABA})
n.CopyModel('dc_generator', 'dc_input', {'amplitude':500.})
n.CopyModel('spike_detector', 'measure')
# create layer E1
E1 = tp.CreateLayer({'rows':N_E1, 'columns':N_E1, 'elements':'iaf_exc_slow', 'extent':[N_E1+1.,N_E1+1.], 'edge_wrap':True})
# create layer I1
I1 = tp.CreateLayer({'rows':N_I, 'columns':N_I, 'elements':'iaf_inh', 'extent':[N_I+1.,N_I+1.], 'edge_wrap':True})
# create input neuron
IN = n.Create('dc_input',N_IN)
M = tp.CreateLayer({'rows':N_E1, 'columns':N_E1, 'elements':'measure', 'extent':[N_E1+1.,N_E1+1.], 'edge_wrap':True})
# define connection dictionary for connections between E1 and E1
cdictE1E1 = {'connection_type':'divergent', 'mask': {'circular': {'radius': N_E1/2.}}, 'weights': {'gaussian': {'p_center':S_E1E1, 'sigma':sigma_E1E1}}, 'allow_autapses':False}
# define connection dictionary for connections between I1 and E1 (from E1 to I1)
cdictI1E1 = {'connection_type':'convergent', 'mask': {'rectangular': {'lower_left':[-(N_I)/2.,-(N_I)/2.], 'upper_right':[(N_I)/2.,(N_I)/2.]}}, 'kernel': c_IE, 'weights':w_I1E1, 'allow_autapses':False} #'kernel': {'constant': c_IE},
# define connection dictionary for connections between I1 and 11
cdictI1I1 = {'connection_type':'convergent', 'mask': {'rectangular': {'lower_left':[-(N_I)/2.,-(N_I)/2.], 'upper_right':[(N_I)/2.,(N_I)/2.]}}, 'kernel': c_II, 'weights':w_I1I1, 'allow_autapses':False}
# define connection dictionary for connections between E1 and I1 (from I1 to E1)
cdictE1I1 = {'connection_type':'divergent', 'mask': {'rectangular': {'lower_left':[-(N_E1)/2.,-(N_E1)/2.], 'upper_right':[(N_E1)/2.,(N_E1)/2.]}}, 'kernel': c_EI, 'weights':w_E1I1, 'allow_autapses':False}
cdictE1M = {'connection_type':'divergent', 'mask': {'circular': {'radius': 1/4.}}, 'weights': 1., 'allow_autapses':False}
# connect Layer E1 with itself
tp.ConnectLayers(E1, E1, cdictE1E1)
# connect Layers E1 and I1 (connections from E1 to I1)
tp.ConnectLayers(E1, I1, cdictI1E1)
# connect Layer I1 with itself
tp.ConnectLayers(I1, I1, cdictI1I1)
# connect Layers E1 and I1 (connections from I1 to E1)
tp.ConnectLayers(I1, E1, cdictE1I1)
# connect input neuron to E0
tp.ConnectLayers(E1, M, cdictE1M)
n.DivergentConnect([IN[0]],tp.FindCenterElement(E1),delay=0.01,weight=1.0)
#fig = tp.PlotLayer(I1)
#ctr = tp.FindCenterElement(I1)
#tp.PlotTargets(ctr, I1, fig=fig, mask={'rectangular': {'lower_left':[-(N_I+1.)/2.,-(N_I+1.)/2.], 'upper_right':[(N_I+1.)/2.,(N_I+1.)/2.]}}, kernel={'gaussian': {'p_center':S_E1E1, 'sigma':sigma_E1E1}})
#pl.show()
t1 = 10
t2 = 10
deltas = np.zeros((t1+t2,N_E1*N_E1))
spikes = np.zeros(N_E1*N_E1)
spikesn = np.zeros(N_E1*N_E1)
n.PrintNetwork(depth=2)
for t in range(t1):
	n.Simulate(simtime)
	for i in range(N_E1*N_E1):
		spikesn[i] = n.GetStatus([M[0]+1+i],'n_events')[0]
	deltas[t] = spikesn-spikes
	spikes = spikesn
pl.figure()
pl.imshow(spikes.reshape((N_E1,N_E1)))
pl.colorbar()
pl.show()
n.SetStatus(n.GetConnections(IN), 'weight',0.)
for t in range(t2):
	n.Simulate(simtime)
	for i in range(N_E1*N_E1):
		spikesn[i] = n.GetStatus([M[0]+1+i],'n_events')[0]
	deltas[t+t1] = spikesn-spikes
	spikes = spikesn
pl.figure()
pl.plot(int(simtime)*np.arange(1,t1+t2+1),np.sum(spikes,axis=1))
pl.show()
pl.figure()
pl.imshow(spikes.reshape((N_E1,N_E1)))
pl.colorbar()
pl.show()
