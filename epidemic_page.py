#-----------------------#
# Import libraries      #
#-----------------------#

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from numba import njit

#-----------------------------#
# Make the style for the page #
#-----------------------------#

#Add title for the page
st.set_page_config(page_title='Simulate an epidemic',
                         page_icon = 'favicon-16x16.png', 
                         layout = 'wide', initial_sidebar_state = 'auto')

#Add a title
st.write('''# Simulate an epidemic 
Using the **SIRS model**, simulate an epidemic using different parameters.
''')

#Sidebar header
st.sidebar.header('User Input')

#Create a function to get the users input
def get_input():
    N = int(st.sidebar.text_input('Total population N (in int)', '10000'))
    tf = float(st.sidebar.text_input('Total time of simulation','100'))
    I0 = float(st.sidebar.text_input('Initial infected population','10'))
    R0 = float(st.sidebar.text_input('Initial recovered population','0'))
    v = float(st.sidebar.text_input('Rate of recovered per period','0.05'))
    beta = float(st.sidebar.text_input('Rate of infected per period','0.00005'))
    gamma = float(st.sidebar.text_input('Rate of recovered to susceptible','0.015'))
    lam = float(st.sidebar.text_input('Rate of births per period','0.0014'))
    mu = float(st.sidebar.text_input('Rate of death per period','0.0025'))
    indexing = int(st.sidebar.text_input('Indexing for the table (in int)','101'))
    return N,tf,I0,R0,v,beta,gamma,lam,mu,indexing

#-------------------------------------#
# Solution                            #
#-------------------------------------#
@njit()
def sirs_bd(N,tf,I0,R0,v,beta,gamma,lam,mu,t,dt):
    S,I,R=IC(N,S0,I0,R0)
    for i in range(0,N-1):
        S[i+1]=S[i]*(1-dt*(beta*I[i]+mu))+dt*(gamma+lam)*R[i]+dt*lam*I[i]
        I[i+1]=I[i]*(1+dt*(beta*S[i]-v-mu))
        R[i+1]=R[i]*(1-dt*(gamma+mu))+dt*v*I[i]
    return S,I,R

@njit()
def IC(N,S0,I0,R0):
    S=np.zeros(N)
    I=np.zeros(N)
    R=np.zeros(N)
    #Put the initial conditon for every system
    S[0]=S0
    I[0]=I0
    R[0]=R0
    return S,I,R

N,tf,I0,R0,v,beta,gamma,lam,mu, indexing= get_input()
S0= N-I0-R0
t=np.linspace(0,tf,N)
dt=tf/N #Timestep


#Set the data to the solution
S,I,R=sirs_bd(N,tf,I0,R0,v,beta,gamma,lam,mu,t,dt)

#--------------------------------#
# Formating data                 #
#--------------------------------#

#Create the arrays
dead_total=np.zeros(N)
dead_period=np.zeros(N)
born=np.zeros(N)

#Asign the initial values
dead_total[0]=mu*(S0+I0+R0)
dead_period[0]=dead_total[0]
born[0]=lam*(N)

#Fullfill the data
for i in range(1,N):
    dead_period[i]=mu*(S[i]+I[i]+R[i])
    dead_total[i]=dead_period[i]+dead_total[i-1]
    born[i]=lam*(S[i]+I[i]+R[i])+born[i-1]

info={
    'Susceptible population': S,
    'Infected population': I,
    'Recovered population': R,
    'From recovered to susceptible': gamma*I,
    'Population born per period': lam*(S+I+R),
    'Population dead per period': dead_period,
    'Total population': S+I+R
}


SIRS_data=pd.DataFrame(data=info, index=t)
SIRS_data=SIRS_data.rename_axis(index='Time (days)')
SIRS_data=SIRS_data.iloc[::indexing,:].astype(int)
SIRS_data.index=SIRS_data.index.astype(int)

#-----------------------------#
# Graph for the solution      #
#-----------------------------#
def plot(S,I,R,t,N,mu):
    #Theme
    theme='plotly_white'
    #layouts
    layout=go.Layout(
        yaxis=dict(title_text='Population',showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
                    #text=['{0:.0f} day'.format(t[i]) for i in range(len(t))]),
        width=1000,
        height=700,
        template=theme,
        hovermode='x'
    )
    fig=go.Figure(layout=layout)
    Sdata=pd.DataFrame({'Susceptible pop.':S, 'Dead in S':mu*S})

    #Splot
    fig.add_trace(go.Scatter(
        x=t,
        y=S,
        line=dict(color='teal'),
        name='Susceptible',
        fill='tozeroy',
        fillcolor='rgba(150,216,216,0.25)',
        hovertemplate=
        '<b>Population</b>: %{y:.0f}'+
        '<br><b>%{text}</b>',
        text=['Dead: {0:.0f}'.format(mu*S[i]) for i in range(len(S))],
    
    ))
    #Iplot
    fig.add_trace(go.Scatter(
        x=t,
        y=I,
        name='Infected',
        fill='tozeroy',
        line=dict(color='crimson'),
        fillcolor='rgba(255,74,50,0.3)',
        hovertemplate=
        '<b>Population</b>: %{y:.0f}'+
        '<br><b>%{text}</b>',
        text=['Dead: {0:.0f}'.format(mu*I[i]) for i in range(len(I))],
    ))
    #Rplot
    fig.add_trace(go.Scatter(
        x=t,
        y=R,
        name='Recovered',
        line=dict(color= 'green'),
        fill='tozeroy',
        fillcolor='rgba(94,191,63,0.3)',
        hovertemplate=
        '<b>Population</b>: %{y:.0f}'+
        '<br><b>%{text}</b>',
        text=['Dead: {0:.0f}'.format(mu*R[i]) for i in range(len(R))],
    ))

    t_vals=[t[i] for i in range(int(N/5),len(t),int(N/5)-1)]
    t_label=['{0:.0f} days'.format(t[i]) for i in range(int(N/5),len(t),int(N/5)-1)]

    fig.update_traces(mode='lines')
    fig.update_layout( hovermode='x')
    fig.update_xaxes(tickvals=t_vals,ticktext=t_label,range=[t[0],t[N-1]],
    hoverformat=',f <b>days</b>')
    fig.update_yaxes(range=[0,N+100])
    st.plotly_chart(fig)

input_graf2=plot(S,I,R,t,N,mu)

#-----------------------------#
# Extra: DataFrame            #
#-----------------------------#

st.write('''## Table of the results
You can check the results of the simulation in the below Dataframe.

If you change the days or total population and doesn't show correctly the data,
in the *User input* section, you need to change the value for the Indexing until it works.
''')

st.table(SIRS_data)

#----------------------------#
# Extra: Model               #
#----------------------------#

st.write('''## More about the SIRS model.
$$
\Large\\begin{array}{c}
\\frac{dS}{dt}=\Lambda(S+I+R)-\\beta SI+\gamma R-\mu S \\\\
\\frac{dI}{dt}=\\beta SI-vI-\mu I \\\\
\\frac{dR}{dt}=vI-\gamma R-\mu R
\end{array}
$$
In this equations, $S$ are the susceptible individuals, $I$ are 
the infected individuals and $R$ are the Recovered individuals. 
$\\beta$ are the rate of individuals per period that become infected
 and $v$ are the rate of individuals per period that become recovered.
With only this values, this is called SIR model. 

We can look that $\\beta$ is multiplied by $S$ and $I$ because 
this rate depends of both individuals, while $v$ only is 
multiplied by $I$. In this model, the total poblation is 
conserved (there are no births or deaths).

We add a parameter $\gamma$ that represent the rate of conversion 
from recovered to susceptible per period. This model is now called SIRS model.


Also, I add a birth and death rate, where $\Lambda$ are the rate of births from 
the total population that resides in that moment 
(that is $S,I$ and $R$). And also, I add $\mu$ that is the 
rate of individuals who died.''')

#------------------------#
# Extra: Contact section #
#------------------------#

st.write('''## More projects & more about me.

Hi! My name is **Ricardo M. Leal Lopez** and thanks for use my web application.

I'm a computational physicist that works with Python, Julia, Matlab and Fortran
to simulate several models, such as heat conduction, the behavior of a wave, etcetera.
Right now i'm on my path to study a graduate course.


If you want to check more projects or want to now more of my habilities/jobs,
you can check it by [clicking here](https://ricardoleal20.github.io/Blog/).
''',unsafe_allow_html=True)