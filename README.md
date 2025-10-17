# red\_cardinal

The `red_cardinal` repository stores all work related to my master's thesis and supporting Jupyter notebooks, as well as a custom-made Python package called `miri_utils/`. The repository serves as both a development environment and a backup, ensuring that all critical code is safely stored and version-controlled. The main focus is on the analysis of dust emission in galaxies at cosmic noon using JWST/MIRI observations from the PRIMER, COSMOS-Web, and COSMOS-3D surveys. Additionally, it includes the `prospector_utils/` library containing all helper scripts regarding the loading and manipulation of Prospector outputs. See overplot_miri.ipynb as a reference.

## 📁 Project Structure

```
red_cardinal/
├── miri_utils/                    # Utilities related to MIRI data preparation and photometry
│   ├── __init__.py
│   ├── astrometry_utils.py        
│   ├── cutout_tools.py
│   ├── photometry_tools.py
│   └── stamp_maker.py
│
├── prospector_utils/              # Utilities related to reconstructing Prospector outputs
│   ├── __init__.py                # and investigating its agreement with MIRI photometry
│   ├── analysis.py        
│   ├── params.py
│   └── plotting.py
│
├── astrometry.ipynb         
├── make_stamps.ipynb
├── overplot_miri.ipynb        
├── photometry.ipynb         
├── produce_cutouts.ipynb
├── prosparams.py       
├── README.md
├── rotate_fits.ipynb
└── webbpsf_tutorial.ipynb
```
## 📄 License

This project is intended for academic use. It will undergo further refinement and will soon be available for public use. If you wish to use it in the meantime, beware of potential library conflicts, depending on the version of Prospector you are using. In the near future, I will compile a list of the exact packages and versions used in these scripts.

## 📫 Contact

For questions or collaborations, feel free to reach out via email or GitHub.

benjaminphilip.collins@studio.unibo.it
