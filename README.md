<h1 align="center">Droplet-culture</h1>



<p align="center">
  <strong>Official implementation of “Droplet culture of tumour organoids”</strong>
</p>

---
<h2>Overview</h2>
<p>The pipeline supports:</p>

<ul>
<li>Droplet diameter pre-estimation based on conditions;</li>
<li>Organoid growth-tracking (Multi-object-tracking)</li>
<li>Organoid time course (Viability changing)</li>
</ul>

---

<h3>Oragnoid growth tracing</h3>

<p align="left">
  <img src="docs/tracking_knocked.gif" alt="Morphi demo" width="400">
</p>


---
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

