---
layout: home
title: Home
landing-title: 'COVID-19 Statistics and Research'
description: 'This website aims to help increase the public’s understanding of the evolving COVID-19.'
image: null
author: 'Miquel Oliver & Xisco Jimenez Forteza'
show_tile: true
---  

<div w3-include-html="./assets/tables/last_update.html"></div>

 
<!-- Main -->
<div id="main">
These curves were obtained using the two models mentioned. You can find an extended explanation <a data-scroll href="#logistic">bellow</a>.
    <div class="image">
        <img src="{% link assets/images/log-modelSigmoid-simulation-linear.png %}" alt="" data-position="center center" width="100%"/>
    </div>
        <div class="row top-buffer1"></div>
<p> In this figure we compare the current number of COVID-19 fatalities to date with the Logistic model shown with color dots, with the multiple projections drawn from the posterior predictive distribution; these projections are shown as faint solid lines (note that the more lines we have the more likely that path will be). We have defined the zeroth time for each country to the day they announced their first fatality record. The vertical grid lines represent important events that may have affected the growth rate such as the separate lockdowns (LD) applied by China, Italy, and Spain. Note that all curves have been drawn from a Logistic model and predict the # fatalities (N) for each country analyzed here.</p>

<div class="image">
        <img src="{% link assets/images/gompertz-modelSigmoid-simulation-linear.png %}" alt="" data-position="center center" width="100%"/>
    </div>
        <div class="row top-buffer1"></div><p>Same information as for the figure above  for the Gompertz model.  Note that all curves  drawn from this model predict the # fatalities (N) few factors larger than for the Logistic model.</p>
</div>
<!--    <div class="image">
        <img src="{% link assets/images/log-modelSigmoid-simulation-log.png %}" alt="" data-position="center center" width="100%"/>
    </div>
    <div class="row top-buffer1"></div>
            <p>This figure shows the same results as the previous one but now we have changed the Total # of deaths axis to a logarithmic scale. This generates a visual straight line at the beginning of the outbreak and it "bends down" as it surpasses the inflection point; this point corresponds to the moment for which the evolution of the growth starts to slow down. Note that our estimates are consistent with the total number of deaths within [2000, 6000] for all countries. These estimates could vary with any relaxation of the measures each country takes.</p>
</div>
>
<!-- Two -->
<h4>Estimated end & the number of fatalities:</h4>
<div class="row">
  <div class="column">
            <img src="{% link assets/images/daystoendlog-model.png %}" alt="" data-position="center center" width="100%"/>
            <p>Marginalized probability distribution for the Logistic model on the outbreak's date,  i.e. when the number of deaths per day tends to zero. As we can see in the figure China has already surpassed that point. Given this model and with current data, this model predicts an earlier flattenning with respect to the Gompertz one. </p>
  </div>
  <div class="column">
        <img src="{% link assets/images/fatalitieslog-model.png %}" alt="" data-position="center center" width="100%"/>
        <p>Marginalized probability distribution on the estimated number of deaths for the Logistic model. The prediction for China agrees with the 3226 cases reported at 17/03/2020. These cases must be taken as a lower limit to the death estimate number. Any non-expected burst of the curve may increase the number of cases. Notice that the # of deaths is smaller than for the Gomertz curve (below), thus defining our optimistic scenario. A strict confinement of the population is therefore essential.</p>
  </div>
</div>

<div class="row">
  <div class="column">
            <img src="{% link assets/images/daystoendgompertz-model.png %}" alt="" data-position="center center" width="100%"/>
            <p>Marginalized probability distribution on the outbreak's date for the Gompertz model. The values for China using this model are accurately reproduced as well. Notice that the # of deaths is larger than for the Logistic curve.</p>
  </div>
  <div class="column">
        <img src="{% link assets/images/fatalitiesgompertz-model.png %}" alt="" data-position="center center" width="100%"/>
        <p>Marginalized probability distribution on the estimated number of deaths for the Gompertz model. Notice that the # of deaths is larger than for the Logistic curve, thus providing our pessimistic scenario.</p>
  </div>
</div>



<section id="one">
    <div class="inner">
        <h3>The Data:</h3>
    </div>
    <p>On the table below we show the COVID-19 data collected for the set of countries with at least one fatality. We show the number of deaths in each country (# of deaths), the number of days since the first official report of death (# days), the growth rate that reflects the percentual increase of the deaths in that day i.e. (Today-Yesterday)/Yesterday x 100 and the growth factor we will discuss more about it in a future section. Finally, the growth factor (GF) is represented as (Today-Yesterday)/(Yesterday-Day before) i.e. the quotient of today's and yesterday's derivatives.</p>
    <div class="row top-buffer"></div>
    <div w3-include-html="./assets/tables/tabledata.html"></div>
    <script>
    includeHTML();
    </script>
</section>


<!-- Two -->
<section id="two" class="spotlights">
    <section>
        <div class="content">
            <img width="350" src="{% link assets/images/pic08.jpg %}" alt="" data-position="center center" />
        </div>
        <div class="content">
            <div class="inner">
                <header class="major">
                    <h3>JHU CSSE repository</h3>
                </header>
            </div>
            <p>The data used in this repository comes from the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) repository devoted to the 2019 Novel Coronavirus Visual Dashboard operated.</p>
            <div class="row">
                <div class="col-md-6">
                    <ul class="actions">
                        <li><a href="https://github.com/CSSEGISandData/COVID-19" class="button">Repository</a></li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <ul class="actions">
                        <li><a href="https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6" class="button">Dashboard</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </section>
</section>


<div class="row top-buffer"></div>

<section id="three">
    <section>
        <div class="inner">
            <header class="major">
                <h3>The death evalution simulations:</h3>
            </header>
        </div>
        <div id="logistic"></div>
        <h3>Exponential & logistic growth</h3>
        <p>A population will grow its size according to a growth rate. In the case of exponential growth, this rate stays the same regardless of the population size, inducing the population to grow faster and faster as it gets larger, without an end.</p>
        <ul>
            <li>In nature, populations can only grow exponentially during some period, but inevitably the growth rate will ultimately be limited for example by the resource availability.</li>
            <li>In logistic growth, the population growth rate gets smaller and smaller as population size approaches a maximum. This maximum is, in essence, a product of overpopulation limiting the population's resources.</li>
            <li>Exponential growth produces a J-shaped curve, while logistic growth produces an S-shaped curve.</li>
            <li>When we read about bending the curve we are talking about using a logarithmic scale to plot the data, in that case, that J-shaped curve becomes a straight line. The moment when this straight line bends downwards we start seeing the limiting factors and we are close to the center of the S-shaped curve, which in this case looks like an inverse J-shape. (Remember this is only a matter of how the data is plotted or shown, it does not affect the data itself).</li>
        </ul>        
        <h5>*Text from: <a href="https://www.khanacademy.org/science/biology/ecology/population-growth-and-regulation/a/exponential-logistic-growth">khanacademy</a>*</h5> 
        <p>In theory, any kind of organism could take over the Earth just by reproducing. For instance, imagine that we started with a single pair of male and female rabbits. If these rabbits and their descendants reproduced at top speed ("like bunnies") for 777 years, without any deaths, we would have enough rabbits to cover the entire state of Rhode Island. And that's not even so impressive – if we used E. coli bacteria instead, we could start with just one bacterium and have enough bacteria to cover the Earth with a 111-foot layer in just 36 hours!</p>

        <p>As you've probably noticed, there isn't a 111-foot layer of bacteria covering the entire Earth (at least, not at my house), nor have bunnies taken possession of Rhode Island. Why, then, don't we see these populations getting as big as they theoretically could? E. coli, rabbits, and all living organisms need specific resources, such as nutrients and suitable environments, in order to survive and reproduce. These resources aren’t unlimited, and a population can only reach a size that match the availability of resources in its local environment.</p>

        <p>Population ecologists use a variety of mathematical methods to model population dynamics (how populations change in size and composition over time). Some of these models represent growth without environmental constraints, while others include "ceilings" determined by limited resources. Mathematical models of populations can be used to accurately describe changes occurring in a population and, importantly, to predict future changes.</p>    
<div class="text-align">
    <img src="{% link assets/images/cartoon_exp_log.png %}" alt="" width="950"/>
    <p>*end of the <a href="https://www.khanacademy.org/science/biology/ecology/population-growth-and-regulation/a/exponential-logistic-growth">khanacademy</a> citation*</p>
</div>
        <p>The figure above shows how the logistic and exponential models are constructed; to underestand them better you can watch the video  <a data-scroll href="#bazinga">"Exponential growth and epidemics"</a> bellow.</p> 
        <p>After reading this text it should be obvious to us that the growth of the virus cannot be exponential indefinitely but it has to flatten at some point. One of these functions is the logistic model, used here to predict the number of deaths. </p>
        <h3>The logistic function:</h3>
        <p>If we solve the equation on the right of the previous figure, we obtain the logistic function. A logistic function or logistic curve is S-shaped. This type of curve is known as a sigmoid and its equation is as follows:</p>
        $$N(t) = \frac{K}{1 + e^{-r(t-t_0)}}.$$
        <ul> 
            <li> $e$ = the natural logarithm base (also known as Euler's number),</li>
            <li> $t_0$ = the $t$-value of the sigmoid where the rate starts to decrease, the midpoint of its evolution and the 'inflexion point' of the sigmoid's curve.</li>
            <li> $K$ =the curve's maximum value; in this case the maximum number of deaths.</li>
            <li> $r$ = the logistic growth rate or steepness of the curve</li>
        </ul>   
        <h3>Simulating possible future scenarios:</h3>
        <p>The logistic model defined above and a nonnegative binomial distribution as likelihood, to obtain the posterior predictive distribution of our model; from which we will sample to generate new data based on our estimated posteriors. (Please do not get disturbed by this, if you want to have a rough idea of the concept behind all this go lower to the video title "The Bayesian Trap" by Veritasium).
        The figures show, considering this dataset and our model, the predicted evolution of the curves that are expected to be observed. Note that the predictions have the uncertainty into account. Meaning that in the cases where few data points are available the uncertainty grows i.e. the spam of the predictions.
        In short, the figures show that given the data and our model, what evolutions are expected to be observed. Note that the predictions have the uncertainty into account. This implies that for the cases where few data points are available this uncertainty grows.</p>      
    </section>
</section>

<!-- Two -->
<div class="row">
  <div class="column" style="background-color:#2d3450;">
    <header class="major">
        <h3>Exponential growth and epidemics:</h3>
    </header>
                    <p>While this video uses COVID-19 as a motivating example, 
                    the main goal is simply a math lesson on exponentials 
                    and logistic curves.</p>
                    <div id="bazinga"></div>
                    <p>by 3Blue1Brown</p>
                    <ul class="actions">
                        <li><a href="https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw" class="button">Go to their channel</a></li>
                    </ul>

  </div>
  <div class="column" style="background-color:#2d3450;">
                    <iframe width="100%" height="100%" src="https://www.youtube.com/embed/Kas0tIxDvrg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
</div>

<div class="row top-buffer"></div>
<div class="inner">
    <header class="major">
        <h3>Posterior parameters:</h3>
    </header>
</div>
<div class="text-fixed-left">
    <p>Here we show the results of the Bayesian inference parameter estimation. Our goal is to show the inferred probability distributions over the model parameters of interest, the probabilities of models and the probability distributions over predicted data.</p>
    <p>This is a universal approach for fitting models to data. We have defined the generative model for the data, the likelihood function, and a prior distribution over the parameters.</p> 
    <p>The following figures show the results:</p>
    <img src="{% link assets/images/parameterslog-model-c1-c2.png %}" alt="" data-position="center center" width="95%"/>
    <p>In this figure, we show the results obtained from the posterior i.e. the most likely scenarios given our current model. We show the results for the growth rate value in terms of the total number of deaths predicted by the logistic curves for all the countries.</p>
</div>
<div class="text-fixed-left">
    <img src="{% link assets/images/parameterslog-model-c3-c1.png %}" alt="" data-position="center center" width="95%"/>
    <p>In this figure we show the results obtained for the 'Inflexion day' in terms of the total growth rate; note that the end of the outbreak is simply the double of the 'Inflexion day'.</p>
</div>
<div class="text-fixed-left">
    <img src="{% link assets/images/parameterslog-model-c3-c2.png %}" alt="" data-position="center center" width="95%"/>
    <p>Figure for the 'Total number of deaths' and the 'Inflexion day'.</p>
</div>
<div style="overflow: auto; width:100%;">
    <div w3-include-html="./assets/tables/bay_summarylog-model.html"></div>
</div>
<div class="row top-buffer"></div>

<div class="row">
  <div class="column" style="background-color:#2d3450;">
    <header class="major">
        <h3>The Bayesian Trap:</h3>
    </header>
                <p>Bayes' theorem explained with examples and implications for life.</p>
                <p>by Veritasium</p>
                    <ul class="actions">
                    <li><a href="https://www.youtube.com/channel/UCHnyfMqiRRG1u-2MsSQLbXA" class="button">Go to their channel</a></li>
                    </ul>

  </div>
  <div class="column" style="background-color:#2d3450;">
            <iframe width="100%" height="100%" src="https://www.youtube.com/embed/R13BD8qKeTg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </div>
</div>

<div class="row top-buffer"></div>


 
<!-- Main -->
<section id="gon">
    <section>
    <div class="inner">
    <header class="major">
        <h2>Gompertz curve model:</h2>
    </header>
</div>
<div id="gompertz"></div>
<p>This curve is an alternative model that could be taken at this point as upper bounds, we have realized that the logistic model tends to fit the inflection point close to the end of the available data, therefore giving most likely a lower bound prediction. We are not going to discuss the origins but simply mention that this curve is a sigmoid function.</p>
<p>Examples of uses for Gompertz curves include:</p>
<ul>
    <li>Modelling of growth of tumors</li>
    <li>Modelling market impact in finance</li>
    <li>Detailing population growth in animals of prey, with regard to predator-prey relationships</li>
    <li>Examining disease spread</li>
    <li>Modelling bacterial cells within a population</li>    
</ul>
$$N(t)=N(0)e^{-e^{-a(t-c)}}$$
<p>where:</p>
<ul>
    <li>$N(0)$ is the initial number of cells/organisms when time is zero</li>
    <li>$a$ denotes the rate of growth</li>
    <li>$b, c$ are positive numbers</li>
    <li>$c$ denotes the displacement across in time</li>
</ul>

<div class="row top-buffer"></div>
<div class="inner">
    <header class="major">
        <h3>Posterior parameters:</h3>
    </header>
</div>
<div class="text-fixed-left">
    <p>Here we show the results of the Bayesian inference parameter estimation for the Gompertz model.</p>
    <p>The following figures show the results:</p>
    <img src="{% link assets/images/parametersgompertz-model-c1-c2.png %}" alt="" data-position="center center" width="95%"/>
    <p>In this figure, we show the results obtained from the posterior i.e. the most likely scenarios given the Gomperz model. We show the results for the growth rate value in terms of the total number of deaths predicted by the logistic curves for all the countries.</p>
</div>
<div class="text-fixed-left">
    <img src="{% link assets/images/parametersgompertz-model-c3-c1.png %}" alt="" data-position="center center" width="95%"/>
    <p>In this figure we show the results obtained for the 'Inflexion day' in terms of the total growth rate; note that the end of the outbreak is simply the double of the 'Inflexion day'.</p>
</div>
<div class="text-fixed-left">
    <img src="{% link assets/images/parametersgompertz-model-c3-c2.png %}" alt="" data-position="center center" width="95%"/>
    <p>Figure for the 'Total number of deaths' and the 'Inflexion day'.</p>
</div>
<div style="overflow: auto; width:100%;">
    <div w3-include-html="./assets/tables/bay_summarygompertz-model.html"></div>
</div>
<div class="row top-buffer"></div>


<!--    <div class="image">
        <img src="{% link assets/images/gompertz-modelSigmoid-simulation-linear.png %}" alt="" data-position="center center" width="100%"/>
    </div>
        <div class="row top-buffer1"></div><p>In this figure we compare the current number of COVID-19 fatalities to date shown with color dots, with the multiple projections drawn from the posterior predictive distribution; these projections are shown as faint solid lines (note that the more lines we have the more likely that path will be). We have defined the zeroth time for each country to the day they announced their first fatality record. The vertical grid lines represent important events that may have affected the growth rate such as the separate lockdowns (LD) applied by China, Italy, and Spain. Note that all curves have been drawn from a Gompertz model and predict the # fatalities (N) for each country analyzed here.</p>

    <div class="image">
        <img src="{% link assets/images/gompertz-modelSigmoid-simulation-log.png %}" alt="" data-position="center center" width="100%"/>
    </div>  
    <div class="row top-buffer1"></div>
            <p>This figure shows the same results as the previous one but now we have changed the Total # of deaths axis to a logarithmic scale.</p>
</section>
</section>

<h3>Gompertz curve model table:</h3>
<div style="overflow: auto; width:100%;">
    <div w3-include-html="./assets/tables/bay_summarygompertz-model.html"></div>
</div>
<!-- 
<div style="overflow: auto; width:100%;">
    <div w3-include-html="./assets/tables/last_bayes_factor.html"></div>
</div>
-->
<div class="row top-buffer"></div>
<!-- Sharingbutton Facebook -->
Share me &#128540;
<div id="share-buttons">
<!-- Email -->
<a href="mailto:?Subject=Simple Share Buttons&amp;Body=I%20saw%20this%20and%20thought%20of%20you!%20 https://simplesharebuttons.com">
    <img src="https://simplesharebuttons.com/images/somacro/email.png" alt="Email" />
</a>

<!-- Facebook -->
<a href="http://www.facebook.com/sharer.php?u=https://simplesharebuttons.com" target="_blank">
    <img src="https://simplesharebuttons.com/images/somacro/facebook.png" alt="Facebook" />
</a>

<!-- Tumblr-->
<a href="http://www.tumblr.com/share/link?url=https://simplesharebuttons.com&amp;title=Simple Share Buttons" target="_blank">
    <img src="https://simplesharebuttons.com/images/somacro/tumblr.png" alt="Tumblr" />
</a>

<!-- Twitter -->
<a href="https://twitter.com/share?url=https://simplesharebuttons.com&amp;text=Simple%20Share%20Buttons&amp;hashtags=simplesharebuttons" target="_blank">
    <img src="https://simplesharebuttons.com/images/somacro/twitter.png" alt="Twitter" />
</a>
</div>

<!-- 
<div class="row top-buffer"></div>
<div class="inner">
    <header class="major">
        <h3>Posterior parameters:</h3>
    </header>
</div>
<div class="text-fixed-left">
    <p>....</p> 
    <p>The following figures show the results:</p>
    <img src="{% link assets/images/Growth-China.png %}" alt="" data-position="center center" width="95%"/>
    <p>.............................................................................</p>
</div>
<div class="text-fixed-left">
    <img src="{% link assets/images/Growth-Iran.png %}" alt="" data-position="center center" width="95%"/>
    <p>.............................................................................</p>
</div>
<div class="text-fixed-left">
    <img src="{% link assets/images/Growth-Italy.png %}" alt="" data-position="center center" width="95%"/>
    <p>.............................................................................</p>
</div>
<div class="text-fixed-left">
    <img src="{% link assets/images/Growth-Spain.png %}" alt="" data-position="center center" width="95%"/>
    <p>.............................................................................</p>
</div>
<div class="row top-buffer"></div> -->
