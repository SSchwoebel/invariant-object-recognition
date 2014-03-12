#invariant object recognition

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
S_E1E1 = 1.0
sigma_E1E1 = 20.
S_E2E1 = 1.0
sigma_E2E1 = 4.
N_E0 = 30
N_E1 = 50
N_E2 = 10
N_I = 10
N_IN = 1
w_I1E1 = 1.0
w_I1I1 = -1.0
w_E1I1 = -1.0
w_GABA = -1.0
w_AMPANMDA = 1.0
w_IN = 0.9
simtime = 30.
xsteps = 20
ysteps = 20

# neuron model with slow excitatory synapses for layer E0 and E2
n.CopyModel('iaf_psc_alpha', 'iaf_exc_slow', {'C_m':C_E, 'V_reset':V_reset, 'V_th':V_th, 'tau_m':tau_m_E, 'tau_syn_ex':tau_NMDA, 'tau_syn_in':tau_GABA})
# neuron model with fast excitatory synapses for layer E1
n.CopyModel('iaf_psc_alpha', 'iaf_exc_fast', {'C_m':C_E, 'V_reset':V_reset, 'V_th':V_th, 'tau_m':tau_m_E, 'tau_syn_ex':tau_AMPA, 'tau_syn_in':tau_GABA})
# neuron model with slow excitatory synapses for layer I1
n.CopyModel('iaf_psc_alpha', 'iaf_inh', {'C_m':C_I, 'V_reset':V_reset, 'V_th':V_th, 'tau_m':tau_m_I, 'tau_syn_ex':tau_NMDA, 'tau_syn_in':tau_GABA})
n.CopyModel('dc_generator', 'dc_input', {'amplitude':500.})
n.CopyModel('stdp_synapse_hom','excitatory-plastic', {'alpha':10.0})
# create layer E0
E0 = tp.CreateLayer({'rows':N_E0, 'columns':N_E0, 'elements':'iaf_exc_fast', 'extent':[N_E0+1.,N_E0+1.], 'edge_wrap':True})
# create layer E1
E1 = tp.CreateLayer({'rows':N_E1, 'columns':N_E1, 'elements':'iaf_exc_slow', 'extent':[N_E1+1.,N_E1+1.], 'edge_wrap':True})
# create layer E2
E2 = tp.CreateLayer({'rows':N_E2, 'columns':N_E2, 'elements':'iaf_exc_fast', 'extent':[N_E2+1.,N_E2+1.], 'edge_wrap':True})
# create layer I1
I1 = tp.CreateLayer({'rows':N_I, 'columns':N_I, 'elements':'iaf_inh', 'extent':[N_I+1.,N_I+1.], 'edge_wrap':True})
# create input neuron
IN = n.Create('dc_input',N_IN)
# define connection dictionary for connections between E1 and E1
cdictE1E1 = {'connection_type':'divergent', 'mask': {'circular': {'radius': N_E1/2.}}, 'weights': {'gaussian': {'p_center':S_E1E1, 'sigma':sigma_E1E1}}, 'allow_autapses':False}
# define connection dictionary for connections between E2 and E1
cdictE2E1 = {'connection_type':'convergent', 'mask': {'circular': {'radius': N_E1/2.}}, 'weights': {'gaussian': {'p_center':S_E2E1, 'sigma':sigma_E2E1}}, 'allow_autapses':False}
# define connection dictionary for connections between I1 and E1 (from E1 to I1)
cdictI1E1 = {'connection_type':'convergent', 'mask': {'rectangular': {'lower_left':[-(N_I)/2.,-(N_I)/2.], 'upper_right':[(N_I)/2.,(N_I)/2.]}}, 'kernel': c_IE, 'weights':w_I1E1, 'allow_autapses':False} #'kernel': {'constant': c_IE},
# define connection dictionary for connections between I1 and 11
cdictI1I1 = {'connection_type':'convergent', 'mask': {'rectangular': {'lower_left':[-(N_I)/2.,-(N_I)/2.], 'upper_right':[(N_I)/2.,(N_I)/2.]}}, 'kernel': c_II, 'weights':w_I1I1, 'allow_autapses':False}
# define connection dictionary for connections between E1 and I1 (from I1 to E1)
cdictE1I1 = {'connection_type':'divergent', 'mask': {'rectangular': {'lower_left':[-(N_E1)/2.,-(N_E1)/2.], 'upper_right':[(N_E1)/2.,(N_E1)/2.]}}, 'kernel': c_EI, 'weights':w_E1I1, 'allow_autapses':False}
# define connection dictionary for connections between E1 and E0 (from E0 to E1)
cdictE1E0 = {'connection_type':'divergent', 'mask': {'rectangular': {'lower_left':[-(N_E1)/2.,-(N_E1)/2.], 'upper_right':[(N_E1)/2.,(N_E1)/2.]}}, 'synapse_model':'excitatory-plastic'}
# connect Layer E1 with itself
tp.ConnectLayers(E1, E1, cdictE1E1)
# connect Layers E1 and E2
tp.ConnectLayers(E1, E2, cdictE2E1)
# connect Layers E1 and I1 (connections from E1 to I1)
tp.ConnectLayers(E1, I1, cdictI1E1)
# connect Layer I1 with itself
tp.ConnectLayers(I1, I1, cdictI1I1)
# connect Layers E1 and I1 (connections from I1 to E1)
tp.ConnectLayers(I1, E1, cdictE1I1)
# connect Layers R0 and E1
tp.ConnectLayers(E0, E1, cdictE1E0)
# connect input neuron to E0
n.DivergentConnect([IN[0]],range(E0[0]+1,E0[0]+1+N_E0*N_E0),delay=0.01,weight=list(np.zeros(N_E0*N_E0)))
#fig = tp.PlotLayer(I1)
#ctr = tp.FindCenterElement(I1)
#tp.PlotTargets(ctr, I1, fig=fig, mask={'rectangular': {'lower_left':[-(N_I+1.)/2.,-(N_I+1.)/2.], 'upper_right':[(N_I+1.)/2.,(N_I+1.)/2.]}}, kernel={'gaussian': {'p_center':S_E1E1, 'sigma':sigma_E1E1}})
#pl.show()
n.PrintNetwork(depth=2)
"""
r = range(E0[0],E1[0])
s = range(IN[0]+1,IN[0]+1+N_IN*N_IN)
conns = n.GetConnections(s, synapse_model='static_synapse')
w = n.GetStatus(conns, 'weight')
pl.figure()
pl.plot(s, w)
pl.show()
"""
"""
r = range(E0[0]+1,E0[0]+1+N_E0*N_E0)
conns = n.GetConnections(r, synapse_model='excitatory-plastic')
w = n.GetStatus(conns, 'weight')
pl.figure()
pl.plot(range(len(w)), w)
pl.show()
"""
for xstep in range(2):
	for ystep in range(2):
		print "xstep "+str(xstep)+", ystep "+str(ystep)
		data = np.loadtxt('blobs/blob['+str(xstep+5)+','+str(ystep+5)+'].txt')
		data = list(data.reshape(data.size))
		conns = n.GetConnections(IN)
		for i in range(N_E0*N_E0):
			n.SetStatus([list(conns[i])], 'weight',2*data[i])
		n.Simulate(simtime)
		"""
		conns = n.GetConnections(IN)
		w = n.GetStatus(conns, 'weight')
		pl.figure()
		pl.imshow(np.array(w).reshape(N_E0,N_E0),cmap='gray')
		pl.show()
		"""
r = range(E0[0]+1,E0[0]+1+N_E0*N_E0)
conns = n.GetConnections(r, synapse_model='excitatory-plastic')
w = n.GetStatus(conns, 'weight')
pl.figure()
pl.plot(range(len(w)), w)
pl.show()
"""
n.Simulate(simtime)
conns = n.GetConnections(r, synapse_model='stdp_synapse_hom')
w = n.GetStatus(conns, 'weight')
pl.figure()
pl.hist(w, bins=100)
pl.show()
"""
