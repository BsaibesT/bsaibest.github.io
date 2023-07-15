---
layout: page
title: DRUMS!
description: DRUMS!
img:
importance: 4
category: fun
---

<!-- I have been playing drums since 2004. -->
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!
DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS! DRUMS!

DRUMS! I LOVE THEM! I PLAY THEM!

Current Gear:\\
Cymbals
{::nomarkdown}<ul><li>18" UFIP Rough Crash</li></ul>{:/}
{::nomarkdown}<ul><li>22" Dream Bliss Ride</li></ul>{:/}
{::nomarkdown}<ul><li>19" Zildjian Crash/Ride (Not sure what it is, but it's old?)</li></ul>{:/}
{::nomarkdown}<ul><li>24" Bosphorous Antique Ride</li></ul>{:/}
{::nomarkdown}<ul><li>15" Istanbul Agop Xist Dark Hats </li></ul>{:/}

Drums\\
{::nomarkdown}<ul><li>Not sure, I found them in a dumpster and fixed them up. </li></ul>{:/}

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}
```html
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
```
{% endraw %}
