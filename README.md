<h1 align="center">
    <img src="https://github.com/WSU-Carbon-Lab/pyref/assets/73567020/f4883d3b-829e-48da-9a66-df50ecf357e5" alt="Pyref logo">
    <br>
</h1>

## Pyref: The blazingly fast X-ray analysis toolkit


Pyref is a X-ray analysis toolkit based on the software libraries [pypxr](https://github.com/usnistgov/P-RSoXR),
[kkcalc](https://github.com/benajamin/kkcalc), and [refnx](https://github.com/refnx/refnx) implemented in rust,
and python, taking advantage of Polars. This toolkit provides the means to analyse X-ray experimental data
=======
Pyref is a X-ray analysis toolkit based on the software libraries [pypxr](https://github.com/usnistgov/P-RSoXR), 
[kkcalc](https://github.com/benajamin/kkcalc), and [refnx](https://github.com/refnx/refnx). This is a work in progress to transition these projects into a more throughput oriented. The end goal is to combine the benefits of rust with the usability of python, taking advantage of Polars. This toolkit provides the means to analyse X-ray experimental data 
for the following experiments.

- X-ray Reflectivity: Tier 1 support
- X-ray Scattering: Tier 2 support
- X-ray Diffraction: Tier 3 support