# General Procedure Outline
Harlan Heilman

## Data Sorting
After the data is collected, the `xrr.FileSorter.main` object manages sorting the collected data into a single processed directory. This Directory will be structured as follows
```
~\home\...\[Insert Sorted Directory Name Here]\[Insert Sample Name Here]\Energy\Pol
```
This allows the fits files to be stored in a simple standardized working directory. Note, the main process in `xrr.FileSorter.main` asks for user input on the sample name. In our experience, working at ALS BL 11.0.1.2, sample naming can be incorrectly logged during beam time, and this work around solves this naming issue. Of course the user can choose to avoid this process and name their samples properly during beam time.

## Raw Data Loading
This processing method makes liberal use of a `pandas` `DataFrame` backend. This allows for easy management of `.fits` file metadata for each image in a reflectivity experiment. In Loading the raw data, we use `astropy` to open each fits file, saving the header data as a single row in a metadata DataFrame. For reflectivity, only some metadata is needed, namely beamline energy, sample theta, and beamline current.

The image data is collected using `astropy`, and stored in a separate DataFrame. This reduces the size of the DataFrame as it is used in future computations, while simplifying the data saving process.

## Reflectivity Computation
To compute the reflectivity, one must locate the beam on the CCD. This process uses a simple maximum finding algorithm to locate the index of the most intense point, before drawing a box around said point for numerical integration. The steps in this peak finding algorithm are:

* Remove the CCD border
* Apply any masking to the image
* Apply a median filter to the image
* Generate the Beam Spot Image
* Generate the Dark Spot Image

With the Beam Spot Image,a dn Dark Spot image, each are numerically summed before the specular reflectivity is computed, i.e.,
$$
    R = I_{beam} - I_{dark},
$$
Where $I_{beam}$ is the beam total intensity in the Beam Spot Image, and $I_{dark}$ the total intensity in the Dark Spot Image. 

After this computation, the scattering vector is computed using values from the metadata,
$$
    Q = \frac{4\pi \epsilon}{\hbar c}\sin(\theta_{2}),
$$
where $Q$ is the scattering vector, $\epsilon$ is the photon energy, and $\theta_2$ is two theta. 

## Normalization
The above computed "Specular Reflectivity" is not the Reflectance spoken about by Fresnel or Cauchy, but is instead the absolute intensity measured by the CCD with a background subtraction to remove some noise that is inherent to the 11.0.1.2 CCD. To instead get the reflectance, we must normalize this computation to the Incident beam intensity. This is done by measuring direct beam intensity before any reflectivity measurements. This direct beam measurement or $I_0$ is then used to normalize future data, i.e., 
$$
    R = \frac{I_{beam} - I_{darl}}{I_0}.
$$
Additionally, measuring the direct beam intensity allows us to gain a better understanding of the uncertainty of future points in the computation. 

## Stitching
The crux of all problems with this experiment is something we will call the order of magnitude problem. Each pixel value on the 11.0.1.2 CCD is represented by an unsigned 16 bit integer in their analog to digital conversion. This means that the maximum value measurable by the CCD is 65536 counts. The noise floor for this CCD is at around 1,000 counts, meaning that there are only 2 orders of magnitude to work with in this experiment. 
<p align="center">
  <img src="refl.png" />
</p>
Often, as seen in the above image, reflectivity data stretches significantly more than two orders of magnitude. The solution to this is stitching. 


The q-range is broken up into several subsets. In each subset, experimental parameters are selected to maximize the intensity over the subset. While you might not be able to detect the beam out to a sample theta of $70^\circ$ when dwelling for the .001 sec needed for izero scan, you certainty can if you wait for 10 min with reduced higher order suppressors, and opened scatter slits. 


