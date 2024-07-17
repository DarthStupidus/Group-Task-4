# Group-Task-4

Can you predict the Premier League?

Group Members - Alex Lee, James Hanson Eleanor Duplock, Mohammad Ali and Essa Bostan

The aim of this project is to create a model that will accurately predict the season ending position of each English Premier League team based upon a variety of factors.  This model was then taken to create an app where a person can then enter in variables and then get a predicted placing.

Creating a model.

The success of this project was dependant on collecting relevant data.  Data was collected from a variety of sources - Kaggle, transfermarkt.co.uk, premierleague.com and footballteamnews.com, and Opta.

Initially the following data was collected:

Season Ending (Year)
Team Name
Squad Value - this was taken from the same point for each season - June 15th.
Wages paid to players
Whether the manager from the previous season was retained for the upcoming season
Wages paid to the Manager
Whether the team captain from the previous season was retained for the upcoming season
Net Summer Transfer Spend
Net Number of transfers i.e. did squad size shrink or grow?
Average Home Attendance
Average Distance to away games
Previous season's Goal Difference
Previous season's Disciplinary Points
Previous season's position
Position bin - for the current season, each club was placed into a position bin.  One bin contained those sides who qualified for European football (Top 6), the Mid Table teams, and those who were in or close to the relegation zone (Bottom 6).

Some initial data cleanp was required - for example with those sides who gained promotion to the EPL the previous season.  Their previous season's position would be for a different league so we assigned the side that finished first in the lower league the previous season a position of 18th, the side that finished second a position of 19th and the side that got promoted via the playoffs a position of 20th - so they would occupy the bottom 3 placings in the EPL.

Data on Manager's salary was sparse and there was no meaningful way to expraploate any of the missing data - so it was not included when running the model.

Some initial cleanup was also required when financial information was recieved on different scales.  For example the Squad Vaue was given as a multiple of millions but the wages paid to players given as the full number.  They were all converted to the same scale.

This data was then combined and uploaded onto postgres for retreival by our model.

Using jupyter notebook, a Python script was created.  The script contained a preprocessing phase for identifying problematic columns and performed.  The script initialises, trains and evaluates a Random Forest model, as this was a classification task, Random Forest was felt to be the most appropriate.

From the initial model, an accuracy of 44% was achieved.

Optimising the model.

To improve accuracy, the important features suggested areas for improvement.  It listed features such as team name, season an average distance to away games being important features.  Team name and season are identifiers as to when a datapoint happens and to whom - rather than being a meaningful datapoint - and hence feature - in iteself.  Therefore these were dropped from our refined model.

Average distance to away games is shown as being important - but after discussion we decided to drop this from the dataset as we are looking to use features that e.g. a football club can control as opposed to accidents of geography, which they have no control over.

Further examination of the Goal Difference for the previous season gave some concern for those clubs who were in the previous season playing in the Championship as opposed to the Premier League.  For example Norwich City would tend to run away with the Championship, score a very high number of goals but when they reached the EPL would struggle to score - as well as concede a lot more.  For a time they were viewed as a "YoYo" club - too good for the Champoinship but not good enough to survive consistently in the EPL.  As it stood, it was felt that these newly promoted sides had too great a goal difference.  To counter this, their goal difference was changed.  The figure that was used was from the most recent season that they were in the EPL i.e. the season that they were relegated from the EPL.  This seemed reasonable as this would also correlate with the statistic generated for league position from the previous season.  Fortunately this data was mainly within the timeframe originally selected.  The only real outlier was Brentford who had not been in England's top division since the late 1940's before this current run - but a figure was still able to be obtained.

We wanted to increase the number of potential features especially as we had dropped some earlier on for our model to use.  The first we added was "Games Played".  Conventional logic is that the more successful clubs play more matches - for example those clubs in the Champion's League play at least 6 extra matches as well as their 38 game commitment by being in the Premier League.  This is on top of domestic cup competitons such as the FA cup and the Carabao cup.  For example, when Manchester City won the treble, they played a total of 61 games.  Thy could have played more - but were knocked out at the Quarter Final stage of the Carabao cup.

It was also noted as a concern for some clubs - Aston Villa qualified for European football but at the season start there were concerns that these guaranteed extra games would affect EPL performance as their squad was not percieved as strong enough to cope with this extra demand.  

Note - fortunately as a fan of Aston Villa this concern was not found to be the case in reality - the legend that is Unai Emery produced a higher EPL finish than the previous season.

Manager stability is seen as an important influencer of performance.  For example Sir Alex Ferguson was in charge of Manchester United for 27 years and they were for the vast majority of that time the most dominant force in English football.  Pep Guardiola has been in charge of Manchester City for the last 8 years and they are now the most dominant side in English football and have won the EPL for the last 4 seasons.  Anecdotally there is the perception that those sides that are in danger of relegation change their manager frequently during the season to try and get a "quick fix".  The theory is that a new manager will give a short term improvement in results so relegation is avoided - and hence the financial implications involved.  For example, when the mighty Aston Villa were relegated unluckily in the 2015/2016 season, there were 4 managers throughout the season.  By introducing this parameter we are replacing the Manager salary feature that we could not include due to a lack of data with another metric that can indicate the influence of a manager upon performance.

Once these new features were added - and the irrelevant ones were dropped, the Random Forset Classifer was run again, and the important features looked for.  Accuracy had increased to 96% and the important features seemed more logical - such as the most important feature being squad value this time round as opposed to average distance to away games.  It just makes more sense as well as the accuracy improving.

A further refinement was made - the model is excellent at predicting the top 6 but not so good at predicting the bottom 6.  The boundary between the bottom 6 and mid table seems to be where the issue seems to be.  It was considered whether whee the problem exactly lies.  Is the model really good at predicting the bottom 3 but not so good at predicting the next 3 league placings?  Is it really good at predicting the bottom 4 but not the next 2 and so on?  

The data was then changed - for example on the original dataset, 6 teams per season were placed in the position bin 0 i.e. the relegation zone.  This was then changed where only 5, then 4, then 3 teams were placed in the position bin 0, with the corrsponding number of teams added to position bin 1 i.e. Mid Table.  This did not have the effect of increasing accuracy further, suggesting that teams can be predicted to be in a certain area but for more specific placings, it is harder to do.






