# red\_cardinal

The `red_cardinal` repository stores all work related to my master's thesis and supporting Jupyter notebooks, as well as a custom made python package called `miri_utils/`. The repository serves as both a development environment and a backup, ensuring that all critical code is safely stored and version-controlled. The main focus is on the analysis of galaxies at cosmic noon using data from the JWST, specifically MIRI and NIRCam observations from the PRIMER and COSMOS-Web surveys, as well as NIRSpec data from the BlueJay survey.

## 📁 Project Structure

```
red_cardinal/
├── miri_utils/              # Utilities related to MIRI data preparation and photometry
│   ├── __init__.py
│   ├── astrometry.py        
│   ├── cutouts.py
│   ├── photometry.py
│   ├── rotate.py
│   └── stamps.py
│
├── astrometry.ipynb         
├── make_stamps.ipynb        
├── photometry.ipynb         
├── produce_cutouts.ipynb    
├── README.md
├── rotate_fits.ipynb
└── webbpsf_tutorial.ipynb

## 🛠️ Development Notes

```
* Make sure to periodically sync this repository to GitHub to avoid data loss.
* Use meaningful commit messages to track changes effectively.
* Test each function locally before pushing to the main branch to maintain code quality.

## 📄 License

```
This project is intended for personal academic use. If you wish to use it for other purposes, please contact me for permission.

## 📫 Contact

```
For questions or collaborations, feel free to reach out via email or GitHub.
benjaminphilip.collins@studio.unibo.it
