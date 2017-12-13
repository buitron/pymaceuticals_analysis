
# Pymaceuticals Inc
## Analysis
* There are two drugs that seem to work significantly better than the rest in controlling the spread and growth of the treated tumor. Those drugs are: Capomulin, and Ramicane.
* All of the drugs seem to control the metastastize property of the treated tumor compared to the placebo.
* There is hardly any variation in each drugs performance since the begining of it's utilization on the subject. The path of outcome is very linear throughout the entire experiment, i.e from day 1 to day 45 the drug Zoniferol has had an average rate of change of about positive 1.16 tumor volume (mm3) growth per 5 days.


```python
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from generatoratron import axis_gen, errorbar_gen
```


```python
csv_path1 = path.join('..','raw_data','clinicaltrial_data.csv')
csv_path2 = path.join('..','raw_data','mouse_drug_data.csv')
```


```python
clinical_df = pd.read_csv(csv_path1)
mouse_df =    pd.read_csv(csv_path2)
```


```python
clinical_mouse_df = clinical_df.merge(mouse_df, how='outer', on='Mouse ID')
clinical_mouse_df.head()
```

| | Mouse ID | Timepoint | Tumor Volume (mm3) | Metastatic Sites | Drug |
|-|----------|-----------|--------------------|------------------|------|
| 0 | b128 | 0 | 45.000000 | 0 | Capomulin |
| 1 | b128 | 5 | 45.651331 | 0 | Capomulin |
| 2 | b128 | 10 | 43.270852 | 0 | Capomulin |
| 3 | b128 | 15 | 43.784893 | 0 | Capomulin |
| 4 | b128 | 20 | 42.731552 | 0 | Capomulin |


## Tumor Response to Treatment


```python
tumor_response_gp_mn = clinical_mouse_df.groupby(by=['Drug',
                                            'Timepoint']).mean()[['Tumor Volume (mm3)']]
tumor_response_gp_mn.head()
```

| | | Tumor Volume (mm3) |
|-|-|--------------------|
| Drug | Timepoint | |
| Capomulin | 0 | 45.000000 |
| | 5 | 44.266086 |
| | 10 | 43.084291 |
| | 15 | 42.064317 |
| | 20 | 40.716325 |


```python
tumor_response_pv_mn = pd.pivot_table(clinical_mouse_df,values='Tumor Volume (mm3)',
                                index=['Timepoint'], columns=['Drug'], aggfunc=np.mean)
tumor_response_pv_mn.head()
```

| Drug | Capomulin | Ceftamin | Infubinol | Ketapril | Naftisol | Placebo | Propriva | Ramicane | Stelasyn | Zoniferol |
|------|-----------|----------|-----------|----------|----------|---------|----------|----------|----------|-----------|
| Timepoint | | | | | | | | | | |
| 0 | 45.000000 | 45.000000 | 45.000000 | 45.000000 | 45.000000 | 45.000000 | 45.000000 | 45.000000 | 45.000000 | 45.000000 |
| 5 | 44.266086 | 46.503051 | 47.062001 | 47.389175 | 46.796098 | 47.125589 | 47.248967 | 43.944859 | 47.527452 | 46.851818 |
| 10 | 43.084291 | 48.285125 | 49.403909 | 49.582269 | 48.694210 | 49.423329 | 49.101541 | 42.531957 | 49.463844 | 48.689881 |
| 15 | 42.064317 | 50.094055 | 51.296397 | 52.399974 | 50.933018 | 51.359742 | 51.067318 | 41.495061 | 51.529409 | 50.779059 |
| 20 | 40.716325 | 52.157049 | 53.197691 | 54.920935 | 53.644087 | 54.364417 | 53.346737 | 40.238325 | 54.067395 | 53.170334 |


```python
tumor_response_gp_se = clinical_mouse_df.groupby(by=['Drug',
                                            'Timepoint']).sem()[['Tumor Volume (mm3)']]
tumor_response_gp_se.head()
```

| | | Tumor Volume (mm3) |
|-|-|--------------------|
| Drug | Timepoint |
| Capomulin | 0 | 0.000000 |
| | 5 | 0.448593 |
| | 10 | 0.702684 |
| | 15 | 0.838617 |
| | 20 | 0.909731 |


```python
tumor_response_pv_se = pd.pivot_table(clinical_mouse_df,values='Tumor Volume (mm3)',
                                index=['Timepoint'], columns=['Drug'], aggfunc=sem)
tumor_response_pv_se.head()
```


| Drug | Capomulin | Ceftamin | Infubinol | Ketapril | Naftisol | Placebo | Propriva | Ramicane | Stelasyn | Zoniferol |
|------|-----------|----------|-----------|----------|----------|---------|----------|----------|----------|-----------|
| Timepoint | | | | | | | | | | |
| 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 0.448593 | 0.164505 | 0.235102 | 0.264819 | 0.202385 | 0.218091 | 0.231708 | 0.482955 | 0.239862 | 0.188950 |
| 10 | 0.702684 | 0.236144 | 0.282346 | 0.357421 | 0.319415 | 0.402064 | 0.376195 | 0.720225 | 0.433678 | 0.263949 |
| 15 | 0.838617 | 0.332053 | 0.357705 | 0.580268 | 0.444378 | 0.614461 | 0.466109 | 0.770432 | 0.493261 | 0.370544 |
| 20 | 0.909731 | 0.359482 | 0.476210 | 0.726484 | 0.595260 | 0.839609 | 0.555181 | 0.786199 | 0.621889 | 0.533182 |


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.448593</td>
      <td>0.164505</td>
      <td>0.235102</td>
      <td>0.264819</td>
      <td>0.202385</td>
      <td>0.218091</td>
      <td>0.231708</td>
      <td>0.482955</td>
      <td>0.239862</td>
      <td>0.188950</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.702684</td>
      <td>0.236144</td>
      <td>0.282346</td>
      <td>0.357421</td>
      <td>0.319415</td>
      <td>0.402064</td>
      <td>0.376195</td>
      <td>0.720225</td>
      <td>0.433678</td>
      <td>0.263949</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.838617</td>
      <td>0.332053</td>
      <td>0.357705</td>
      <td>0.580268</td>
      <td>0.444378</td>
      <td>0.614461</td>
      <td>0.466109</td>
      <td>0.770432</td>
      <td>0.493261</td>
      <td>0.370544</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.909731</td>
      <td>0.359482</td>
      <td>0.476210</td>
      <td>0.726484</td>
      <td>0.595260</td>
      <td>0.839609</td>
      <td>0.555181</td>
      <td>0.786199</td>
      <td>0.621889</td>
      <td>0.533182</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = list(tumor_response_pv_mn.index)
y_axis = axis_gen(tumor_response_pv_mn)
sem_axis = axis_gen(tumor_response_pv_se)
```


```python
plt.figure(figsize=(15,10))
plt.ylim(tumor_response_pv_mn.values.min() - 10,tumor_response_pv_mn.values.max() + 10)
plt.xlim(0,max(x_axis))

# function call to generate plots
errorbar_gen(y_axis, sem_axis, tumor_response_pv_mn, x_axis, sem_axis)

plt.title('Tumor Response to Treatment', size=18)
plt.xlabel('Time (Days)', size=14)
plt.ylabel('Tumor Volume (mm3)', size=14)
plt.legend(fontsize=12, edgecolor='black', loc=2, fancybox=True, numpoints=2)
plt.grid(ls='dashed')
plt.show()
```


![drug tumor response scatterplot](../images/output_11_0.png)


## Metastatic Response to Treatment


```python
metastatic_response_gp_mn = clinical_mouse_df.groupby(by=['Drug',
                                            'Timepoint']).mean()[['Metastatic Sites']]
metastatic_response_gp_mn.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Capomulin</th>
      <th>0</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.160000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.320000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.652174</td>
    </tr>
  </tbody>
</table>
</div>




```python
metastatic_response_pv_mn = pd.pivot_table(clinical_mouse_df,values='Metastatic Sites',
                                index=['Timepoint'], columns=['Drug'], aggfunc=np.mean)
metastatic_response_pv_mn.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.160000</td>
      <td>0.380952</td>
      <td>0.280000</td>
      <td>0.304348</td>
      <td>0.260870</td>
      <td>0.375000</td>
      <td>0.320000</td>
      <td>0.120000</td>
      <td>0.240000</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.320000</td>
      <td>0.600000</td>
      <td>0.666667</td>
      <td>0.590909</td>
      <td>0.523810</td>
      <td>0.833333</td>
      <td>0.565217</td>
      <td>0.250000</td>
      <td>0.478261</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.375000</td>
      <td>0.789474</td>
      <td>0.904762</td>
      <td>0.842105</td>
      <td>0.857143</td>
      <td>1.250000</td>
      <td>0.764706</td>
      <td>0.333333</td>
      <td>0.782609</td>
      <td>0.809524</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.652174</td>
      <td>1.111111</td>
      <td>1.050000</td>
      <td>1.210526</td>
      <td>1.150000</td>
      <td>1.526316</td>
      <td>1.000000</td>
      <td>0.347826</td>
      <td>0.952381</td>
      <td>1.294118</td>
    </tr>
  </tbody>
</table>
</div>




```python
metastatic_response_gp_se = clinical_mouse_df.groupby(by=['Drug',
                                            'Timepoint']).sem()[['Metastatic Sites']]
metastatic_response_gp_se.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Capomulin</th>
      <th>0</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.074833</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.125433</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.132048</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.161621</td>
    </tr>
  </tbody>
</table>
</div>




```python
metastatic_response_pv_se = pd.pivot_table(clinical_mouse_df,values='Metastatic Sites',
                                index=['Timepoint'], columns=['Drug'], aggfunc=sem)
metastatic_response_pv_se.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.074833</td>
      <td>0.108588</td>
      <td>0.091652</td>
      <td>0.098100</td>
      <td>0.093618</td>
      <td>0.100947</td>
      <td>0.095219</td>
      <td>0.066332</td>
      <td>0.087178</td>
      <td>0.077709</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.125433</td>
      <td>0.152177</td>
      <td>0.159364</td>
      <td>0.142018</td>
      <td>0.163577</td>
      <td>0.115261</td>
      <td>0.105690</td>
      <td>0.090289</td>
      <td>0.123672</td>
      <td>0.109109</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.132048</td>
      <td>0.180625</td>
      <td>0.194015</td>
      <td>0.191381</td>
      <td>0.158651</td>
      <td>0.190221</td>
      <td>0.136377</td>
      <td>0.115261</td>
      <td>0.153439</td>
      <td>0.111677</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.161621</td>
      <td>0.241034</td>
      <td>0.234801</td>
      <td>0.236680</td>
      <td>0.181731</td>
      <td>0.234064</td>
      <td>0.171499</td>
      <td>0.119430</td>
      <td>0.200905</td>
      <td>0.166378</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = list(metastatic_response_pv_mn.index)
y_axis = axis_gen(metastatic_response_pv_mn)
sem_axis = axis_gen(metastatic_response_pv_se)
```


```python
plt.figure(figsize=(15,10))
plt.ylim(metastatic_response_pv_mn.values.min(),metastatic_response_pv_mn.values.max() + .5)
plt.xlim(0,max(x_axis))

# function call to generate plots
errorbar_gen(y_axis, sem_axis, metastatic_response_pv_mn, x_axis, sem_axis)

plt.title('Metastatic Spread During Treatment', size=18)
plt.xlabel('Treatment Duration (Days)', size=14)
plt.ylabel('Metastatic Sites', size=14)
plt.legend(fontsize=12, edgecolor='black', loc=2, fancybox=True, numpoints=2)
plt.grid(ls='dashed')
plt.show()
```


![metastatic spread to treatment scatterplot](../images/output_18_0.png)


## Survival Rates


```python
survival_rate_gp_ct = clinical_mouse_df.groupby(by=['Drug',
                                            'Timepoint']).count()[['Mouse ID']].rename(columns={'Mouse ID': 'Mouse Count'})
survival_rate_gp_ct.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Mouse Count</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Capomulin</th>
      <th>0</th>
      <td>25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
    </tr>
    <tr>
      <th>20</th>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
survival_rate_pv_ct = pd.pivot_table(clinical_mouse_df,values='Mouse ID',
                                index=['Timepoint'], columns=['Drug'], aggfunc=np.ma.count)
survival_rate_pv_ct.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>21</td>
      <td>25</td>
      <td>23</td>
      <td>23</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>24</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>21</td>
      <td>24</td>
      <td>23</td>
      <td>24</td>
      <td>23</td>
      <td>22</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
      <td>19</td>
      <td>21</td>
      <td>19</td>
      <td>21</td>
      <td>20</td>
      <td>17</td>
      <td>24</td>
      <td>23</td>
      <td>21</td>
    </tr>
    <tr>
      <th>20</th>
      <td>23</td>
      <td>18</td>
      <td>20</td>
      <td>19</td>
      <td>20</td>
      <td>19</td>
      <td>17</td>
      <td>23</td>
      <td>21</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_axis = list(survival_rate_pv_ct.index)
y_axis = [list(survival_rate_pv_ct[element].divide(survival_rate_pv_ct[element].max()).multiply(100))
           for element in survival_rate_pv_ct.columns]
sem_axis = np.linspace(0,0,100, dtype=int).reshape(10,10)

```


```python
plt.figure(figsize=(15,10))
plt.ylim(20,100)
plt.xlim(0,max(x_axis))

# function call to generate plots
errorbar_gen(y_axis, sem_axis, survival_rate_pv_ct, x_axis, sem_axis)

plt.title('Survival During Treatment', size=18)
plt.xlabel('Time (Days)', size=14)
plt.ylabel('Survival Rate (%)', size=14)
plt.legend(fontsize=12, edgecolor='black', loc=3, fancybox=True, numpoints=2)
plt.grid(ls='dashed')
plt.show()
```


![survival rate to drug scatterplot](../images/output_23_0.png)


## Summary Bar Graph


```python
tumor_change_over_treatment = ((tumor_response_pv_mn.iloc[-1] - tumor_response_pv_mn.iloc[0])
                               /tumor_response_pv_mn.iloc[0])*100
tumor_change_over_treatment
```




    Drug
    Capomulin   -19.475303
    Ceftamin     42.516492
    Infubinol    46.123472
    Ketapril     57.028795
    Naftisol     53.923347
    Placebo      51.297960
    Propriva     47.241175
    Ramicane    -22.320900
    Stelasyn     52.085134
    Zoniferol    46.579751
    dtype: float64




```python
x_axis = np.arange(0,len(tumor_response_pv_mn),1)
y_axis = np.array(list(tumor_change_over_treatment))

mask_good = y_axis < 0
mask_bad = y_axis >= 0

labels = list(tumor_response_pv_mn.columns)
```


```python
fig, ax = plt.subplots(figsize=(15,10))

ax.bar(x_axis[mask_good], y_axis[mask_good], width=1, edgecolor=['black']*len(x_axis[mask_good]), color='green', zorder=3)
ax.bar(x_axis[mask_bad], y_axis[mask_bad], width=1, edgecolor=['black']*len(x_axis[mask_bad]), color='red', zorder=3)
ax.set_xticks(list(x_axis))
ax.set_xticklabels(labels=labels)


ax.set_title('Tumor Change Over 45 Day Treatment', fontdict={'fontsize':18})
ax.set_ylabel('% Tumor Volume Change', fontdict={'fontsize':14})

for bar in ax.patches:
    ax.text(bar.get_x()+.12, bar.get_height()*.92, "{:.2f}%".format(bar.get_height()),
            color='white', fontdict={'size':16, 'weight':'heavy'})

ax.set_ylim(-25,60)
ax.set_xlim(-.5,9.5)
ax.grid(ls='dashed', zorder=0)

plt.show()
```


![tumor change with drug bargraph](../images/output_27_0.png)

