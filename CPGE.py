#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from scipy.special import expit

class PrefixedFormatter(argparse.HelpFormatter):
    def format_help(self):
        lines = super().format_help().split("\n")
        lines.insert(0, "")
        lines.append("\n")
        return "\n\t".join(lines)

parser = argparse.ArgumentParser(
    description="Compute circular photogalvanic effect (CPGE) from DFT outputs.",
    formatter_class=PrefixedFormatter,
)
parser.add_argument("--dmu", type=float, help="Change in mu relative to mu (if available) or VBM of DFT calculation [eV]", required=True)
parser.add_argument("--n_blocks", type=int, help="Read eigenvals and momenta files with prefix 1, 2, 3, ... nblocks", required=True)
parser.add_argument("--n_bands", type=int, help="Number of bands to use for CPGE spectrum", required=True)
parser.add_argument("--s_band", type=int, help="Starting of band to use for CPGE spectrum", default=0)
parser.add_argument("--domega", type=float, help="Bin size for histogram [eV]", default=0.1)
parser.add_argument("--omegaMax", type=float, help="Frequency cutoff of CPGE signal [eV]", default=10.0)
parser.add_argument("--T", type=float, help="Temperature at which to compute CPGE [K]", default=298)
parser.add_argument("--prefix", type=str, help="Directory where scf totalE.* files exist", default='.')
parser.add_argument("--omegaAnalysis", type=float, nargs='*', help="One or more frequencies to perform transition analysis [eV]", default = [])
args = parser.parse_args()

def main():
	
    eV = 1/27.21138505
    Kelvin = 1./3.157750E5

    dmu = args.dmu*eV
    n_blocks = args.n_blocks
    n_bands = args.n_bands
    s_band = args.s_band
    domega = args.domega*eV
    omegaMax = args.omegaMax*eV
    kT = args.T*Kelvin
    t_prefix = args.prefix
    decomp = True
    omega_analysis_eV = args.omegaAnalysis
    omega_analysis = [omega_eV * eV for omega_eV in omega_analysis_eV]

    # Read mu, spin weight and lattice vectors from totalE.out:
    mu = np.nan
    R = np.zeros((3,3))
    refLine = -10
    Rdone = False
    initDone = False
    for iLine,line in enumerate(open(f'{t_prefix}/totalE.out', encoding="utf-8", errors="ignore")):
    	if line.startswith('Initialization completed'):
    		initDone = True
    	if initDone and line.find('FillingsUpdate:')>=0:
    		mu = float(line.split()[2])
    		mu_method = "Fermi level"
    	if line.startswith('spintype'):
    		n_spinor = (1 if line.split()[1] in {"no-spin", "z-spin"} else 2)
    		spinWeight = (2. if line.split()[1] == "no-spin" else 1.)
    	if (not initDone) and line.startswith('nElectrons:'):
    		nElectrons = float(line.split()[1])
    		nBandsDFT = int(line.split()[3])
    		nStates = int(line.split()[5])
    		nValence = (int(np.round(nElectrons)) * n_spinor) // 2 #number of valence bands (in SOC mode)
    		Edft = np.reshape(np.fromfile(f'{t_prefix}/totalE.eigenvals'), (nStates, nBandsDFT))
    		mu = np.max(Edft[:,:nValence]) #VBM
    		mu_method = "VBM"
    	if line.find('Initializing the Grid') >= 0:
    		refLine = iLine
    	if not Rdone:
    		rowNum = iLine - (refLine+2)
    		if rowNum>=0 and rowNum<3:
    			R[rowNum,:] = np.array([ float(x) for x in line.split()[1:-1] ])
    		if rowNum==3:
    			Rdone = True
    assert not np.isnan(mu)
    cellVolume = np.abs(np.linalg.det(R))
    print(f'OmegaMax: ({omegaMax})')
    print(f'mu: {mu} ({mu_method})')
    print(f'dmu: {dmu}')
    print(f'T: {kT}')
    print(f'cellVolume: {cellVolume}')
    print(f'spinWeight: {spinWeight}')
    print(f'n_blocks: {n_blocks}')
    print(f'n_bands: {n_bands}')
    print(f's_band: {s_band}')
    total_bands = n_bands-s_band
    print(f'Total Bands: {total_bands}')

    #Read symmetries
    if os.path.isfile(f'{t_prefix}/totalE.sym'):
        symData = np.loadtxt(f'{t_prefix}/totalE.sym').reshape((-1,4,3))
        symTrans = symData[:,3] #translations in lattice coordinates
        sym = symData[:,:3] #rotations in lattice coordinates
        symCart = np.einsum('ab,sbc,cd->sad', R, sym, np.linalg.inv(R))
        doSym = True
    else:
        print('Not Including Symmetry')
        doSym = False

    eps = leviCivita()
 
    #start collection

    beta = Histogram(0.1*eV, omegaMax, domega, 9)
    CPGE_analysis = [None] * len(omega_analysis)  # transition analysis for each frequency in omega_analysis
    if decomp:
        jdos = Histogram(0.1*eV, omegaMax, domega, 1)
        Delta = Histogram(0.1*eV, omegaMax, domega, 3)
        rr = Histogram(0.1*eV, omegaMax, domega, 3)
        
    n_k_tot = 0  # total k-points collected
    for i_block in range(1, n_blocks+1):     
        if not os.path.isfile(f"./block_{i_block}/bandstruct.R"):
            print(f"\n\tSkipping block {i_block}\n")
            continue

        E, V, R, proj = read_block(f"./block_{i_block}", n_bands,s_band,omega_analysis )

        if proj is not None:
            shell_names, proj_shell = proj.by_shell()

        if decomp:
            kpt = read_kpt(f"./block_{i_block}/sampling.kpoints.in")

        prefac_per_k = np.pi / cellVolume
        
        n_k_tot += len(E)
        print(f'\nBlock {i_block} ({len(E)} kpoints):', end='', flush=True)
        for ik, (Ek, Vk, Rk) in enumerate(zip(E, V, R)):
        
            # Fermi Distribution 
            Fk = expit((mu + dmu - Ek)/kT)
        
            # Fermi Occupation Difference 
            deltaFk = Fk[None, :] - Fk[:, None]
            # Energy Difference
            omega = Ek[:, None] - Ek[None, :]
            # Velocity Difference
            deltaVk = Vk[:, None, :] - Vk[None, :, :]
            # Select Events
            bSel = np.where(np.logical_and(
            	np.logical_and(omega > domega, omega < omegaMax),
            	np.abs(deltaFk) > 1E-6, 
            ))
        
            if len(bSel[0]) == 0:
                #print('None')
                continue
            weights = (prefac_per_k * deltaFk[bSel])[:, None]  # extra dimenion to broadcast with components below
        
            # --- single band-pair dimension now:
            omega = omega[bSel]
            deltaFk = deltaFk[bSel]
            Rknm, Rkmn = sym_Rknm(Rk, bSel)
            deltaVk = deltaVk[bSel]
 
            betaTemp = np.einsum('jkl, n, ni, nk, nl -> nij', eps, deltaFk, deltaVk, Rknm, Rkmn, optimize='optimal') 
            if decomp:
                ksign = np.sign(kpt[ik])
                DeltaTemp = np.einsum('n, ni, i -> ni', deltaFk, deltaVk, ksign, optimize='optimal') 
                rrTemp = np.einsum('jkl, n, nk, nl, j -> nj', eps, deltaFk, Rknm, Rkmn, ksign, optimize='optimal') 
                
            if doSym:
                betaTemp = point_group_symmetrize(betaTemp, symCart)
            betaTemp = betaTemp.reshape(betaTemp.shape[0],-1).imag
            # Histogram:
            beta.add_events(omega, weights * betaTemp)
            if decomp:
                jdos.add_events(omega, weights)
                Delta.add_events(omega, weights * DeltaTemp.reshape(DeltaTemp.shape[0],-1).real)
                rr.add_events(omega, weights * rrTemp.reshape(rrTemp.shape[0],-1).imag)
                
            # Projections:
            if proj is not None:
                for i_omega, omega_i in enumerate(omega_analysis):
                    sel_i = np.where(np.abs(omega - omega_i) < 3 * domega)[0]
                    w_omega = np.exp(-0.5*((omega[sel_i] - omega_i)/domega)**2) / (
                        domega * np.sqrt(2*np.pi)
                    )  # Gaussian weights for contributing to omega_i
                    proj_e = proj_shell[ik, bSel[0][sel_i]]
                    proj_h = proj_shell[ik, bSel[1][sel_i]]
                    contrib = np.einsum(
                        "ai, a, ax, ay -> ixy",
                        betaTemp[sel_i], w_omega, proj_e, proj_h
                    )
                    if CPGE_analysis[i_omega] is None:
                        CPGE_analysis[i_omega] = contrib
                    else:
                        CPGE_analysis[i_omega] += contrib
        
        
    print('\n done.', flush=True)
    print(f'\n\t{n_k_tot} kpoints in total\n') 
    component_names = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
    header = "omega[eV] " + " ".join(f"Beta_{comp}" for comp in component_names)
        
    np.savetxt(f'{n_blocks}-{s_band}-{total_bands}CPGE-beta_ij.dat', np.hstack((beta.bins[:, None]/eV, beta.hist/n_k_tot)), header=header)

    if decomp:
        header = "omega[eV] " + "JDOS"
            
        np.savetxt(f'{n_blocks}-{s_band}-{total_bands}CPGE-JDOS.dat', np.hstack((Delta.bins[:, None]/eV, jdos.hist/n_k_tot)), header=header)
        header = "omega[eV] " + " ".join(f"Delta_{comp}" for comp in ["x", "y", "z"])
            
        np.savetxt(f'{n_blocks}-{s_band}-{total_bands}CPGE-Delta_ij.dat', np.hstack((Delta.bins[:, None]/eV, Delta.hist/n_k_tot)), header=header)

        header = "omega[eV] " + " ".join(f"RR_{comp}" for comp in ["x", "y", "z"])
        
        np.savetxt(f'{n_blocks}-{s_band}-{total_bands}CPGE-rr_ij.dat', np.hstack((rr.bins[:, None]/eV, rr.hist/n_k_tot)), header=header)

    # Save projections:
    if proj is not None:
        for omega_eV_i, CPGE_i in zip(omega_analysis_eV, CPGE_analysis):
            CPGE_i *= (1./n_k_tot)  # normalize for total blocks computed
            prefix = f"CPGE_analysis_{omega_eV_i:.1f}eV"
            print(f"Saving {prefix}")
            np.savetxt(
                f'{prefix}.dat',
                CPGE_i.reshape(-1, len(shell_names)),  # (component * e-orb) x h-orb
                header=" ".join(shell_names)
            )
            for comp, CPGE_comp in zip(component_names, CPGE_i):
                max_abs = np.abs(CPGE_comp).max()
                plt.matshow(CPGE_comp.T, cmap="RdBu", vmin=-max_abs, vmax=max_abs)
                tick_pos = np.arange(len(shell_names))
                tick_labels = [name.replace("_", " ") for name in shell_names]
                plt.xticks(tick_pos, tick_labels, rotation="vertical")
                plt.yticks(tick_pos, tick_labels)
                plt.xlabel("Conduction orbital")
                plt.ylabel("Valence orbital")
                plt.colorbar(label=r"$\beta_{\mathrm{" + comp + r"}}$ contribution")
                plt.title(r"Net $\beta_{\mathrm{" + comp + r"}}$ = " + f"{CPGE_comp.sum():.3g}")
                plt.savefig(f"{prefix}_{comp}.png", bbox_inches="tight",dpi=300)

def read_kpt(fin):
    
    with open(fin, 'r') as f:
        nskip = 0
        for line in f:
            if line.startswith('kpoint '):
                break
            else:
                nskip += 1
    kpoints = np.loadtxt(fin, skiprows=nskip, usecols=(1,2,3))
    return kpoints

def read_block(prefix, n_bands,s_band, omega_analysis):
    E = np.fromfile(f'{prefix}/bandstruct.eigenvals', dtype=np.float64)
    v = np.fromfile(f"{prefix}/bandstruct.velocities",np.float64)
    R = np.fromfile(f'{prefix}/bandstruct.R', dtype=np.complex128)
    # Determine size:
    n_bands_file = len(R) // (3 * len(E))
    n_k = len(E) // n_bands_file
    assert n_bands <= n_bands_file
    # Reshape and select requested band count:
    E = E.reshape(n_k, n_bands_file)[:, s_band:n_bands]
    v = v.reshape(n_k, n_bands_file, 3)[:, s_band:n_bands,:]
    R = R.reshape(n_k, 3, n_bands_file, n_bands_file).swapaxes(1, 3)[:, s_band:n_bands, s_band:n_bands, :]  # fix matrix order and put k, b1, b2, dir
    if len(omega_analysis) > 0:
        proj = Projections(f'{prefix}/bandstruct.bandProjections', n_bands)
    else:
        proj = None
    return E, v, R, proj

def sym_Rknm(Rk, bSel):
    
    Rknm = Rk[bSel]    
    tRkmn = Rk[bSel[::-1]]
    cRknm = np.conj(tRkmn)
    
    aveRknm = np.average([Rknm,cRknm],axis=0)
    
    return aveRknm, np.conj(aveRknm)
	

def leviCivita():
	"""Get Levi-Civita tensor"""
	eps = np.zeros((3, 3, 3))
	eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = +1.
	eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1.
	return eps

def det_kweight(nkpts,gbox):
    '''
       Get kweight
       nkpts : number of kpoints
       gbox : fractional reciprocal vectors for k point samling
    '''
    det = np.linalg.det(gbox)
    return det/nkpts

def point_group_symmetrize(M, symCart):
    return np.einsum('aij,six,sjy->axy', M, symCart, symCart) * (1./len(symCart))

class Histogram:

	def __init__(self, x_min, x_max, dx, n_w):
		"""Initiate histogram with bins arange(x_min, x_max, dx) with n_w weight channels."""
		self.x_min = x_min
		self.x_max = x_max
		self.dx_inv = 1./dx
		self.n_w = n_w
		self.bins = np.arange(x_min, x_max, dx)
		self.hist = np.zeros((len(self.bins), n_w))
		self.n_intervals = len(self.bins) - 1
	
	def add_events(self, x, w):
		"""Add contributions from x (array of length N) with weights w (N x n_w array)."""
		x_frac = (x - self.x_min) * self.dx_inv  # fractional coordinate
		i = np.floor(x_frac).astype(int)
		# Select range of collection:
		sel = np.where(np.logical_and(i >= 0, i < self.n_intervals))
		i = i[sel]
		t = (x_frac[sel] - i)[:, None]  # add dimension to broadcast with n_w weights below
		w_by_dx = w[sel] * self.dx_inv
		# Histogram:
			
		np.add.at(self.hist, i, (1.-t) * w_by_dx)
		np.add.at(self.hist, i + 1, t * w_by_dx)

class Projections:
    """Band projections, along with meta-data."""
    n_k: int  #: number of k-points in data
    n_bands: int  #: number of bands
    n_proj: int  #: total number of projectors
    species: list[str]  #: symbols for ionic species
    n_atoms: list[int]  #: number of atoms per species
    n_orbitals: list[int]  #: number of orbitals per atom for each species
    n_shells: list[list[int]]  #: number of shells for each l, for each species
    data: np.ndarray  #: n_k x n_bands x n_proj array of projections

    def __init__(
        self, fname: str, n_bands: int = 0, normalize: bool = True
    ) -> None:
        """Read projections, truncating to specified number of bands if nonzero.
        If normalize is True, set relative projections on atomic orbitals,
        so that the probabilities on atomic orbitals will add to 1."""
        for i_line, line in enumerate(open(fname)):
            tokens = line.split()
            if i_line == 0:
                self.n_k = int(tokens[0])
                n_bands_file = int(tokens[2])
                if n_bands:
                    assert n_bands <= n_bands_file
                    self.n_bands = n_bands
                else:
                    self.n_bands = n_bands_file
                self.n_proj = int(tokens[4])
                n_species = int(tokens[6])
                self.species = []
                self.n_atoms = []
                self.n_orbitals = []
                self.n_shells = []
                self.data = np.zeros((self.n_k, self.n_bands, self.n_proj))
            elif i_line >= 2:
                if i_line < n_species+2:
                    self.species.append(tokens[0])
                    self.n_atoms.append(int(tokens[1]))
                    self.n_orbitals.append(int(tokens[2]))
                    self.n_shells.append([int(token) for token in tokens[4:]])
                else:
                    i_k, i_band = divmod(
                        i_line - (n_species + 3), n_bands_file + 1
                    )
                    if (0 <= i_band < self.n_bands) and i_k < self.n_k:
                        proj_cur = np.array([float(tok) for tok in tokens])
                        if normalize:
                            proj_cur *= 1.0 / proj_cur.sum()
                        self.data[i_k, i_band] = proj_cur

    def by_shell(self) -> tuple[list[str], np.ndarray]:
        "Retrieve projections summed over atoms and m, keeping species, n, l"
        n_shells_tot = sum(sum(n_shells_sp) for n_shells_sp in self.n_shells)
        l_names = "spdf"
        shell_names = []
        proj_shell = np.zeros((self.n_k, self.n_bands, n_shells_tot))
        n_proj_prev = 0
        n_shells_prev = 0
        for specie, n_atoms_sp, n_orbitals_sp, n_shells_sp in zip(
            self.species, self.n_atoms, self.n_orbitals, self.n_shells
        ):
            # Sum projections on all atoms of this species:
            n_proj_sp = n_atoms_sp * n_orbitals_sp
            n_proj_next = n_proj_prev + n_proj_sp
            proj_sp = self.data[..., n_proj_prev : n_proj_next].reshape(
                self.n_k, self.n_bands, n_atoms_sp, n_orbitals_sp
            ).sum(axis=2)  # k, b, orbitals (atom summed out)
            n_proj_prev = n_proj_next

            # Collect contributions by shell:
            n_m_tot = sum(
                ((2 * l + 1) * n_shells_l)
                for l, n_shells_l in enumerate(n_shells_sp)
            )
            n_spinor = n_orbitals_sp // n_m_tot
            assert n_spinor in (1, 2)
            n_orbitals_prev = 0
            for l, n_shells_l in enumerate(n_shells_sp):
                n_orbitals_l = (2 * l + 1) * n_spinor
                for i_shell in range(n_shells_l):
                    shell_names.append(f"{specie}_{l_names[l]}")
                    # Sum projetcions over m, s:
                    n_orbitals_next = n_orbitals_prev + n_orbitals_l
                    proj_shell[..., n_shells_prev] = proj_sp[
                        ..., n_orbitals_prev : n_orbitals_next
                    ].sum(axis=-1)
                    n_orbitals_prev = n_orbitals_next
                    n_shells_prev += 1
            assert n_orbitals_prev == n_orbitals_sp
        assert n_proj_prev == self.n_proj
        assert n_shells_prev == n_shells_tot
        return shell_names, proj_shell


if __name__ == "__main__":
	np.set_printoptions(linewidth=200)
	main()
