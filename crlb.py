import numpy as np
from matplotlib import pyplot as plt

from matplotlib import rc 
rc('text', usetex=True)
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


"""
Description:
Compute CRLB for ideal waveguide
for VLA and planar endfire array

Date:
3/29/2021

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


D = 216
freq = 100
c=1500
#zr = np.array([36])

def get_kz(D, freq, c):
    """
    Ideal waveguide kz
    """
    k = 2*np.pi*freq/c
    m = 1
    kz = m*np.pi/D
    kzs = []
    while kz*kz < k*k:
        kzs.append(kz)
        m += 1
        kz = m*np.pi/D
    return np.array(kzs)


def get_kr(D, freq, c, kzs):
    """
    Get kr for ideal waveguide with kzs
    """
    k = 2*np.pi*freq/c
    krs = np.sqrt(k*k-np.square(kzs))
    return krs

def get_Az(kzs, zi, z0):
    arg1= kzs*zi
    arg2 = kzs*z0
    Az = np.sin(arg1)*np.sin(arg2)
    return Az

def get_dAzdz(kzs, zi, z0):
    arg1= kzs*zi
    arg2 = kzs*z0
    dAzdz = kzs*np.sin(arg1)*np.cos(arg2)
    return dAzdz

def get_ri(zr, num_synth_els, r0, ship_dr):
    ri = np.zeros((1, zr.size*num_synth_els))
    for i in range(num_synth_els):
        ri[0, i*zr.size:(i+1)*zr.size] = r0 + i*ship_dr
    return ri

def get_zi(zr, num_synth_els):
    zi = np.zeros((zr.size * num_synth_els,1))
    for i in range(num_synth_els):
        zi[i*zr.size:(i+1)*zr.size,0] = zr
    return zi

def get_Br(krs, ri):
    return np.exp(complex(0,1)*krs*ri)/np.sqrt(krs*ri)


def get_dBrdr(krs, ri):
    term1 = complex(0,1)*krs*np.exp(complex(0,1)*krs*ri)/np.sqrt(krs*ri)
    term2 = -0.5*np.exp(complex(0,1)*krs*ri)/np.sqrt(krs*ri)/ri
    return term1+term2

def get_phi(r0, z0, zr, num_synth_els, ship_dr):
    """
    Compute field received at synthetic planar array
    formed from VLA with zr and num_synth_els synthetic
    staves spaced at ship_dr
    """
    zi = get_zi(zr, num_synth_els)
    kzs = get_kz(D, freq, c)
    krs = get_kr(D, freq,c, kzs)
    kzs = kzs.reshape(1, kzs.size)
    krs = krs.reshape(krs.size,1)
    Az = get_Az(kzs, zi, z0)
    ri = get_ri(zr, num_synth_els, r0, ship_dr)
    Br = get_Br(krs, ri)
    phi = np.einsum('ij,ji -> i', Az, Br)
    return phi

def get_dphi_dz(r0, z0,zr, num_synth_els, ship_dr):
    zi = get_zi(zr, num_synth_els)
    kzs = get_kz(D, freq, c)
    krs = get_kr(D, freq,c, kzs)
    kzs = kzs.reshape(1, kzs.size)
    krs = krs.reshape(krs.size,1)
    dAzdz = get_dAzdz(kzs, zi, z0)
    ri = get_ri(zr, num_synth_els, r0, ship_dr)
    Br = get_Br(krs, ri)
    dphidz = np.einsum('ij,ji -> i', dAzdz, Br)
    return dphidz

def get_dphi_dr(r0, z0, zr, num_synth_els, ship_dr):    
    zi = get_zi(zr, num_synth_els)
    kzs = get_kz(D, freq, c)
    krs = get_kr(D, freq,c, kzs)
    kzs = kzs.reshape(1, kzs.size)
    krs = krs.reshape(krs.size,1)
    Az = get_Az(kzs, zi, z0)
    ri = get_ri(zr, num_synth_els, r0, ship_dr)
    dBrdr = get_dBrdr(krs, ri)
    dphidr = np.einsum('ij,ji -> i', Az, dBrdr)
    return dphidr
    

def test_derivs():
    r0 = 5000
    z0 = 50
    num_synth_els = 2
    ship_dr = 0.01
    zr = np.array([20])
    phi = get_phi(r0, z0, zr, num_synth_els, ship_dr)
    dphi = phi[1]-phi[0]
    dphidr = get_dphi_dr(r0, z0, zr, 1, ship_dr)
    print(dphi)
    print(dphidr*ship_dr)


    dz = 0.01
    zr = np.array([20])
    phi1 = get_phi(r0, z0, zr, 1, ship_dr)
    phi2 = get_phi(r0, z0+dz, zr, 1, ship_dr)
    dphi = phi2-phi1
    dphidz = get_dphi_dz(r0, z0, np.array([20]), 1, ship_dr)
    print(dphi)
    print(dphidz*dz)


def get_K_inv(snr_db, r0, z0, zr, num_synth_els, ship_dr):
    phi = get_phi(r0, z0, zr, num_synth_els, ship_dr)
    """ Get avg power of signal """
    phi = phi.reshape(phi.size,1)
    sig_pow = np.mean(np.square(abs(phi)), axis=0)
    """ Get noise var """
    noise_var = sig_pow/np.power(10, snr_db/10)
    term1 = np.identity(phi.size)/noise_var

    term2 = -1/noise_var*phi*(phi.T.conj())/(1 + phi.T.conj()@phi*1/noise_var)/noise_var
    K_inv = term1+term2
    K = noise_var*np.identity(phi.size) + phi@phi.T.conj()
    return K_inv   

def get_dK_dr(r0, z0, zr, num_synth_els, ship_dr):
    phi = get_phi(r0, z0, zr, num_synth_els, ship_dr)
    dphidr = get_dphi_dr(r0, z0, zr, num_synth_els, ship_dr)
    phi = phi.reshape(phi.size,1)
    dphidr =dphidr.reshape(dphidr.size,1)
    return dphidr@phi.T.conj() + phi@(dphidr.T.conj())

def get_dK_dz(r0, z0, zr, num_synth_els, ship_dr):
    phi = get_phi(r0, z0, zr, num_synth_els, ship_dr)
    dphidz = get_dphi_dz(r0, z0, zr, num_synth_els, ship_dr)
    phi = phi.reshape(phi.size,1)
    dphidz =dphidz.reshape(dphidz.size,1)
    return dphidz@phi.T.conj() + phi@(dphidz.T.conj())

def test_dK():
    r0 = 5000
    zs = 36
    zr = np.linspace(50, 80, 20)
    num_synth_els = 1
    ship_dr = 10
    phi = get_phi(r0, z0, zr, num_synth_els, ship_dr)
    phi = phi.reshape(phi.size,1)
    K1 = phi@phi.T.conj()

    dr = 0.01 
    phi = get_phi(r0+dr, z0, zr, num_synth_els, ship_dr)
    phi = phi.reshape(phi.size,1)
    K2 = phi@phi.T.conj()
    dK = K2 - K1
   
    dKdr = get_dK_dr(r0, zs, zr, num_synth_els, ship_dr)
    delta_K = dKdr*dr

    diff = delta_K-dK

    phi = get_phi(r0, z0, zr, num_synth_els, ship_dr)
    phi = phi.reshape(phi.size,1)
    K1 = phi@phi.T.conj()

    dz = 0.001
    phi = get_phi(r0, z0+dz, zr, num_synth_els, ship_dr)
    phi = phi.reshape(phi.size,1)
    K2 = phi@phi.T.conj()
    dK = K2 - K1
   
    dKdz = get_dK_dz(r0, zs, zr, num_synth_els, ship_dr)
    delta_K = dKdz*dz

    diff = delta_K-dK

def get_J(snr_db, r0, zs, zr, num_synth_els, ship_dr):
    K_inv = get_K_inv(snr_db, r0, zs, zr, num_synth_els, ship_dr)
    dKdr = get_dK_dr(r0, zs, zr, num_synth_els, ship_dr)
    dKdz = get_dK_dz(r0, zs, zr, num_synth_els, ship_dr)
    J = np.zeros((2,2), dtype=np.complex128)
    J[0,0] = np.trace(K_inv@dKdr@K_inv@dKdr)
    J[0,1] = np.trace(K_inv@dKdr@K_inv@dKdz)
    J[1,0] = np.trace(K_inv@dKdz@K_inv@dKdr)
    J[1,1] = np.trace(K_inv@dKdz@K_inv@dKdz)
    return J

def get_vars(num_synth_els,snr_dbs, ship_dr):
    r0 = 5000
    z0 = 50
    #ship_dr = 26*2.5
    snr_db = 10
    zr= np.linspace(100, 200, 21)
    #get_K_inv(snr_db, r0, z0, zr, num_synth_els, ship_dr)

    #test_dK()

    r_vars=[]
    z_vars=[]
    
    for snr_db in snr_dbs:
        J = get_J(snr_db, r0, z0, zr, num_synth_els, ship_dr).real
        Jinv = np.linalg.inv(J)
        r_min_var = Jinv[0,0]
        z_min_var = Jinv[1,1]
        r_vars.append(r_min_var)
        z_vars.append(z_min_var)
    
    return r_vars, z_vars
    
    
if __name__ == '__main__':


    snr_dbs =np.linspace(-20, 5, 26)
    fig_name = '/home/hunter/research/coherent_matched_field/paper/pics/crlb.png'


    ship_dr = 25*2.5
    
    #fig, axes = plt.subplots(2,1, sharex='col')
    fig, axis = plt.subplots(1,1)
    #axes[0].grid()
    #axes[1].grid()
    axis.grid()
    colors=['r','b', 'k']
    linestyles=['-', '-.', '--']
    markers = ['*', 'x', '+']
    for i, num_synth_els in enumerate([1, 5, 10]):
        r_vars, z_vars = get_vars(num_synth_els, snr_dbs, ship_dr)
        #axes[0].plot(snr_dbs, r_vars, color=colors[i], marker='+')
        #axes[1].plot(snr_dbs, z_vars, color=colors[i], marker='+')
        axis.plot(snr_dbs, r_vars, color=colors[i], marker=markers[i], linestyle=linestyles[i])


    #axes[0].set_ylim([0, 20])
    #axes[1].set_ylim([0, 5])
    #axes[0].set_xlim([-20, 5])
    axis.set_ylim([0, 20])
    axis.set_xlim([-20, 5])

    #axes[0].text(-19.5, .8, 'a)', fontsize=15, color='k')
    #axes[1].text(-19.5, .2, 'b)', fontsize=15, color='k')
    #axes[1].set_xlabel('Input SNR (dB)',fontsize=12)
    #axes[0].set_ylabel('Range variance (m)', fontsize=15)
    #axes[1].set_ylabel('Depth variance (m)', fontsize=15)
    axis.text(-19.5, .8, 'a)', fontsize=15, color='k')
    #axes[1].text(-19.5, .2, 'b)', fontsize=15, color='k')
    #axes[1].set_xlabel('Input SNR (dB)',fontsize=12)
    axis.set_xlabel('Input SNR (dB)',fontsize=15)
    axis.set_ylabel('Range variance (m)', fontsize=15)
    #axes[1].set_ylabel('Depth variance (m)', fontsize=15)

    #fig.text(0.07, 0.5, 'Variance (m)', va='center', rotation='vertical', fontsize=12)

    plt.legend(['No synth els', '$N_{syn}=5$', '$N_{syn}=10$'])
    plt.savefig(fig_name, dpi=500)

    plt.show()
    #get_J(snr_db, r0, z0, zr, num_synth_els, ship_dr)

    #test_derivs()




    #plt.plot(phi)
    #plt.show()
    #phi = phi.reshape(zr.size, num_synth_els, order='F')
    #phi_tl = 10*np.log10(np.square(abs(phi)) / np.max(np.square(abs(phi))))
    #print(phi_tl.shape)
    #plt.plot(rvals, phi_tl[0,:])

    
    


