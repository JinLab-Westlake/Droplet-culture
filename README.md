<h1 align="center">Droplet-culture</h1>



<p align="center">
  <strong>Official implementation of “Droplet culture of tumour organoids”</strong>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/Paper-Under%20Review-lightgrey" alt="Paper Status">
</p>


The pipeline supports:

<ul>
<li>Droplet diameter pre-estimation based on conditions;</li>
<li>Organoid growth-tracking (Multi-object-tracking)</li>
<li>Organoid time course (Viability changing)</li>
</ul>

</p>




<h2>Overview</h2>


<h4> 1. Drop diameter prediction </h4>
<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Parameter</th>
      <th>Value (Example)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Input</b></td>
      <td>Target diameter (<code>Dp_target</code>)</td>
      <td>153.3364 μm</td>
    </tr>
    <tr>
      <td>BME concentration (<code>conc_new</code>)</td>
      <td>90%</td>
    </tr>
    <tr>
      <td>Container diameter (<code>Dc_new</code>)</td>
      <td>27.3 mm</td>
    </tr>
    <tr>
      <td><b>Output</b></td>
      <td>Predicted rotational speed (<code>N</code>)</td>
      <td><b>≈ 1147.68 RPM</b></td>
    </tr>
  </tbody>
</table>


<h4>2. Oragnoid growth tracing</h4>

<p align="left">
  <img src="docs/tracking_knocked.gif" alt="Morphi demo" width="500">
</p>


<h4>3. Organoid time course </h4>

<h2>Installation</h2>

<p>
  This project uses the environment provided by
  <a href="https://github.com/JinLab-Westlake/Morphi-models">Morphi-models</a>.
</p>

<h3>1. Install Morphi-models</h3>

<pre><code>git clone https://github.com/JinLab-Westlake/Morphi-models.git
cd Morphi-models
bash install.sh
</code></pre>

<p>
  This will create a Conda environment named <code>morphi</code>
  and install all required dependencies.
</p>

<h3>2. Activate the environment</h3>

<pre><code>conda activate morphi
</code></pre>

<h3>3. Clone this repository</h3>

<pre><code>git clone https://github.com/JinLab-Westlake/Droplet-culture.git;
cd Droplet-culture;
</code></pre>

<p>
  No additional dependencies are required. Once the
  <code>morphi</code> environment is activated, the code can be run directly.
</p>

<pre><code>python &lt;your_script.py&gt;
</code></pre>

