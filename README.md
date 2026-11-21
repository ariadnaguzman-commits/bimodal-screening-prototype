# Bimodal Experimental Prototype: Electronic Nose & Automated Cervical Cytology

**Course:** Instrumentación Biomédica (Group 3020, Semester 2027-1) — Facultad de Ciencias, UNAM
**Team:** Arturo Arroyo Nuñez, Paola Canela Lupercio, Ariadna Paola Guzmán Morales
**Supervisors:** Dr. José Eduardo Chairez Veloz, Dr. Alejandro Villarreal López, Misael Saúl Ávila López

## Overview
Signal processing, computer vision, machine learning pipelines, and decoupled visualization interface (Streamlit) for an experimental biomedical instrumentation prototype:
- **Module A (`src/enose/`):** Time-series filtering, baseline normalization ($R_s/R_0$), environmental drift compensation, dynamic feature extraction, and gas classification.
- **Module B (`src/cytology/`):** Cell segmentation (CIELAB + Otsu + morphology) validated with Dice/IoU, morphometric/GLCM feature extraction, and SVM / Random Forest classification on SIPaKMeD.
