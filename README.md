# Can you predict the Premier League?

Group Members - Alex Lee, James Hanson, Eleanor Duplock, Mohammad Ali, Essa Bostan

## Overview

To create a model that will accurately predict the season-ending position of each English Premier League team based on a variety of factors.  This model was then used to create a Flask app where a user can enter variables and get a predicted placement from three "Position Bins" representing Danger of Relegation, Mid-Table Safety, and European Football. This tool could be used by Premier League directors to analyze where best to spend money to ensure their desired results with a high level of accuracy based on historical data.

## Creating a model.

The success of this project was dependent on collecting relevant data to purpose-build a dataset for our specific needs.  Data was collected from a variety of sources - Kaggle, transfermarkt.co.uk, premierleague.com, footballteamnews.com, and Opta.

Initially, the following data was collected:

- Season Ending (Year)
- Team Name
- Squad Value - this was taken from the same point for each season - June 15th.
- Wages paid to players
- Whether the manager from the previous season was retained for the upcoming season
- Wages paid to the Manager
- Whether the team captain from the previous season was retained for the upcoming season
- Net Summer Transfer Spend
- Net Number of transfers i.e. did squad size shrink or grow?
- Average Home Attendance
- Average Distance to away games
- Previous season's Goal Difference
- Previous season's Disciplinary Points
- Previous season's position

And a position bin was added- for the current season, each club was placed into a position bin.  One bin contained those sides who qualified for European football (Top 6), the Mid Table teams, and those who were in or close to the relegation zone (Bottom 6).

Some initial data cleanup was required - for example with those sides who gained promotion to the EPL the previous season.  Their previous season's position would be for a different league so we assigned the side that finished first in the lower league the previous season a position of 18th, the side that finished second a position of 19th, and the side that got promoted via the playoffs a position of 20th - so they would occupy the bottom 3 placings in the EPL.

Data on Managers' salaries was sparse and there was no meaningful way to extrapolate any of the missing data - so it was not included when running the model.

Some initial cleanup was also required when financial information was received on different scales.  For example, the Squad Vaue was given as a multiple of millions but the wages paid to players were given as the full number.  They were all converted to the same scale.

This data was then combined and uploaded onto a Postgres SQL server for retrieval by our model.

Using Jupyter Notebook, a Python script was created.  The script contained a preprocessing phase for identifying problematic columns and performed.  The script then initializes, trains, and evaluates a Random Forest model, as this was a classification task, Random Forest was felt to be the most appropriate.

From the initial model, an accuracy of 44% was achieved.

## Optimising the model

To improve accuracy, the important features suggested areas for improvement.  It listed features such as team name, season, and average distance to away games being important features.  Team name and season are identifiers as to when a datapoint happens and to whom - rather than being a meaningful datapoint - and hence feature - in itself.  Therefore these were dropped from our refined model.

Average distance to away games is shown as being important - but after discussion, we decided to drop this from the dataset as we are looking to use features that a football club can control as opposed to (e.g.) accidents of geography, which they have no control over.

Further examination of the Goal Difference for the previous season gave some concern for those clubs who were in the previous season playing in the Championship as opposed to the Premier League.  For example, Norwich City would tend to run away with the Championship and score a very high number of goals but when they reached the EPL would struggle to score - as well as concede a lot more.  For a time they were viewed as a "YoYo" club - too good for the Championship but not good enough to survive consistently in the EPL.  As it stood, it was felt that these newly promoted sides had too great a goal difference.  To counter this, their goal difference was changed.  The figure that was used was from the most recent season that they were in the EPL i.e. the season that they were relegated from the EPL.  This seemed reasonable as this would also correlate with the statistic generated for league position from the previous season.  Fortunately, this data was mainly within the timeframe originally selected.  The only real outlier was Brentford who had not been in England's top division since the late 1940s before this current run - but a figure was still able to be obtained.

We wanted to increase the number of potential features especially as we had dropped some earlier on for our model to use.  The first we added was "Games Played", to take into account the fatigue that teams build with busier schedules, then "Number of Managers in Season" to take into account the impact of repeated managerial changes on teams.

Once these new features were added - and the irrelevant ones were dropped, the Random Forest Classifier was run again, and the important features looked for.  Accuracy had increased to 96% and the important features seemed more logical - such as the most important feature being squad value this time round as opposed to average distance to away games.  It just makes more sense as well as the accuracy improving.

A further refinement was made - the model is excellent at predicting the top 6 but not so good at predicting the bottom 6.  The boundary between the bottom 6 and mid-table seems to be where the issue seems to be.  It was considered where the problem exactly lies.  Is the model good at predicting the bottom 3 but not so good at predicting the next 3 league placings?  Is it good at predicting the bottom 4 but not the next 2 and so on?  

The data was then changed - for example on the original dataset, 6 teams per season were placed in the position bin 0 i.e. the relegation zone.  This was then changed where only 5, then 4, then 3 teams were placed in the position bin 0, with the corresponding number of teams added to position bin 1 i.e. Mid Table.  This did not have the effect of increasing accuracy further, suggesting that teams can be predicted to be in a certain area but for more specific placings, it is harder to do.

Attempts were made to refine the Random Forest model.  Hyperparameter tuning and feature selection were used.  GridSearchCV was used to find any optimal hyperparameters such as changing the number of trees was performed.  Recursive Feature Elimination was used to identify and keep only the most relevant features.

For GridSearchCV, it successfully fit 1620 different model configurations (324 parameter combinations, each evaluated with 5-fold cross-validation) without any error messages.

Feature selection identified an unlimited depth, no maximum features, with a minimum sample leaf of 1, a minimal sample split of 2, with 100 trees in the forest.  However, after this tuning and feature selection, the model performance gave an accuracy of 92%.  It still predicted those who would finish in the European places, but accuracy dropped slightly with the other two bins.  

While the tuning process was successful in finding a set of parameters, the resulting model does not outperform the original. This suggests that the original model was already well-tuned for the dataset we used.  In short, we got lucky the first time.

As this was a categorization task, Random Forest seemed to be the most appropriate model to use.  We did test other models - using a neural network, a Support Vector Machine, and a gradient boosting one.  None of these three alternatives gave a higher accuracy percentage than the Random Forest model that we currently had.

After the model was established, we decided to use it as a basis for a predictor app.  This was created where parameters can be changed by using sliders and then a prediction as to which bin their theoretical team would fall into.

## Building the User Interface

For the User Interface, we used Flask with HTML renders, also utilizing Javascript and CSS for ease of use. We drafted an initial UX design, which can be found in the Development folder, and worked as a team to create a UX/UI that was close to the original design. Some initially planned features, such as the drop-down menu to select a team, which would have called to a server and pulled previous season data automatically, were dropped due to time constraints.

The app allows a user to input values using sliders and checkboxes to ensure that inputs are within reasonable bounds, the user can then click "Predict" and instantly receive a prediction for the season outcome based on our Machine Learning Model, which had been imported to the app using the Joblib python library. This app, while simple, can predict the final position of a Premier League football team with 92% accuracy based on factors that can be influenced by a football club over the off-season, which is invaluable information for any stakeholder in the multi-billion pound football industry.

## Expansion

As it stands this model and app have a range of potential uses, from being a bit of fun for fans to being a useful business tool for Premier League directors. The Premier League is often affected by money spent, not just how much but also where. At a club level this app could be used to test potential spending plans to see just what impact that could have on the outcome of the next season.

The app could also potentially be expanded by adding other elements into the data set. There are so many statistics within the control of clubs in the Premier League, that there are likely other variables that could be calculated to add more factors into the model.

Beyond that, it would need more data collection and testing, but could a similar model be applied to other football leagues - at home with other english leagues and abroad with, for example, the Serie A in Italy? Even beyond that could similar models be applied to other team league sports?

