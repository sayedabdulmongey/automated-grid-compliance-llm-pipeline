# Automated Grid Compliance LLM Pipeline

This project aims to build an end-to-end MLOps pipeline for fine-tuning a Small Language Model (SLM) for the energy sector, specifically focusing on technical compliance for EV charging and grid standards.

## Project Resources

The following documents are used as the primary knowledge base for the model:

1. **ENA Engineering Recommendation G99 (Issue 2)**: [Link](<https://dcode.org.uk/assets/250307ena-erec-g99-issue-2-(2025).pdf>)
   - _Role_: The technical "Bible" for generation and export limits.
2. **UK Power Networks: Electric Vehicle Connections (EDS 08-5050)**: [Link](https://media.umbraco.io/uk-power-networks/baik5mop/eds-08-5050-electric-vehicle-connections.pdf)
   - _Role_: Mandatory DNO rules for connection, earthing, and charging safety.
3. **SP Energy Networks (SPEN): Connecting your EV fleet**: [Link](https://www.spenergynetworks.co.uk/userfiles/file/Connecting%20your%20EV%20fleet%20V3.pdf)
   - _Role_: Commercial workflow, planning, and project estimation.

## Setup

Data should be placed in `data/raw/` with these exact names:

- `G99_Issue_2.pdf`
- `UKPN_EDS_08_5050.pdf`
- `SPEN_EV_Fleet_Guide.pdf`



# ADD this two commands in the begining to avoid any errors !! 
sudo apt-get update
sudo apt-get install -y tesseract-ocr libgl1