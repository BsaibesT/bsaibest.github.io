---
layout: page
title: Cylinder Alignment
description: Do you need a cylinder aligned with a plane?
img: assets/img/ComsolSim.png
importance: 2
category: work
---
For a cylinder to be a viable test mass in a short range interaction measurement, we need the ability to align it with the source mass. Capacitance can be used as a non-contact method to align a cylindrical test mass with a planar source mass.

<div class="row">
    <div class="col-lg">
        {% include figure.html path="assets/img/ComsolSim.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Simulation of the experimental geometry done in COMSOL.
</div>
Experimental data can be mapped to simulated data to extract the initial separation and angle between a plane and a cylinder. Fitting the experimental data to the simulation requires four fitting parameters: initial height \\(h_0\\), initial angle \\(\theta_0\\), radius of rotation \\(r\\), and offset \\(\delta\\).

<div class="row">
    <div class="col-sm">
        {% include figure.html path="assets/img/experimental_diagram.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Sketch of the experimental geometry.
</div>

The separation and angle are \$$h = h_0 + r(1-\cos(\theta)) - \delta\sin(\theta) + \Delta h$\$ \$$\theta = \theta_0 + \Delta\theta\$$

\\(\Delta h\\) and \\(\Delta\theta\\) are known experimentally. 
Fitting the experimental data to the simulation data is shown below.

<div class="row">
    <div class="col-lg">
        {% include figure.html path="assets/img/refined_fit.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Experimental data (red dots) fitted to the simulation data.
</div>