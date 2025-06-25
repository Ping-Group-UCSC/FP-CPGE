# FP-CPGE
Computes Circular Photogalvanic Effect from first-principles as follows:
```math
\beta_{ij} = \frac{e^3 \pi}{N_{\boldsymbol{k}}V\hbar} \epsilon_{jkl}\sum_{\boldsymbol{k},n,m}f^{\boldsymbol{k}}_{nm}\Delta^i_{\boldsymbol{k},nm}r^k_{\boldsymbol{k},nm}r^l_{\boldsymbol{k},mn}\delta(\hbar\omega-E_{\boldsymbol{k},nm})
```

To compute CPGE first compute energies, velocity and postition matrix elements using JDFTx (other code interfaces are currently not available). Then use the postprocessing script as follows:

	usage: CPGE.py [-h] --dmu DMU --n_blocks N_BLOCKS --n_bands N_BANDS [--s_band S_BAND] [--domega DOMEGA]
	               [--omegaMax OMEGAMAX] [--T T] [--prefix PREFIX] [--omegaAnalysis [OMEGAANALYSIS ...]]
	
	Compute circular photogalvanic effect (CPGE) from DFT outputs.
	
	options:
	  -h, --help            show this help message and exit
	  --dmu DMU             Change in mu relative to mu (if available) or VBM of DFT calculation [eV]
	  --n_blocks N_BLOCKS   Read eigenvals and momenta files with prefix 1, 2, 3, ... nblocks
	  --n_bands N_BANDS     Number of bands to use for CPGE spectrum
	  --s_band S_BAND       Starting of band to use for CPGE spectrum
	  --domega DOMEGA       Bin size for histogram [eV]
	  --omegaMax OMEGAMAX   Frequency cutoff of CPGE signal [eV]
	  --T T                 Temperature at which to compute CPGE [K]
	  --prefix PREFIX       Directory where scf totalE.* files exist
	  --omegaAnalysis [OMEGAANALYSIS ...]
	                        One or more frequencies to perform transition analysis [eV]

