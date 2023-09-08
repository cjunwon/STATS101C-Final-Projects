# Load packages
library(tidyverse)
library(tidymodels)
library(tidyselect)
library(recipes)
library(caret)
library(baguette)
library(rpart)
library(e1071)
library(stacks)
library(xgboost)
library(lightgbm)
library(bonsai)
library(kknn)
library(kernlab)
library(gridExtra)
library(ggridges)
 

   
# Load train and test data
train <- read_csv('train.csv')
test <- read_csv('test.csv')

# Cross-validation data
set.seed(1)
train_folds <- vfold_cv(train, v = 10, strata = percent_dem)
 

# EDA (add later)

   
# Identifying skewed columns:

racial_groups <- train %>%
  select(x0034e:x0039e, -x0036e, x0044e, x0051e)

racial_groups_tidy <- racial_groups %>%
  pivot_longer(cols = x0034e:x0051e, names_to = "race", values_to = "population")

# Calculate the IQR for each race group
racial_groups_tidy <- racial_groups_tidy %>%
  group_by(race) %>%
  mutate(
    Q1 = quantile(population, 0.25),
    Q3 = quantile(population, 0.75),
    IQR = Q3 - Q1
  ) %>%
  ungroup()

# Define a threshold to determine outliers (e.g., 1.5 times the IQR)
outlier_threshold <- 1.5

# Remove outliers
racial_groups_tidy <- racial_groups_tidy %>%
  filter(population >= (Q1 - outlier_threshold * IQR) & population <= (Q3 + outlier_threshold * IQR))

# Create the density ridgeline plot with outliers removed
ggplot(racial_groups_tidy, aes(x = population, y = race, fill = race)) +
  geom_density_ridges() +
  labs(x = "Population", y = "Racial group", title = "Distribution of sample racial groups (outliers removed)") +
  theme(text = element_text(size = 11),
        plot.title = element_text(size = 12),
        axis.title.x = element_text(hjust = 0.5),
        axis.title.y = element_text(hjust = 0.5)) +
  scale_y_discrete(name = "Racial Group",
                   labels = c("One race", "Two or more races", "White", "Black or African American", "American Indian and Alaska Native", "Asian", "Other Asian")) +
  guides(fill = "none")
 

   
# Box plot for the relationship between total votes and percent_Dem
ggplot(train, aes(x = as_factor(x2013_code), y = percent_dem)) +
  geom_boxplot(fill = rainbow(6)) +
  labs(x = "Urban/Rural code used by CDC (1 is most urban 6 is most rural)", y = "Percent of voters who voted Biden", title = "Distribution of Percent of Biden Voters by Urban/Rural Code") +
  theme(text = element_text(size = 12),
        plot.title = element_text(size = 13))
 

   
# Scatterplot sample to show that columns such as population groups require transformations

## scatter plot for the relationship between total votes and percent_Dem
hisp_lat <- ggplot(train, aes(x = total_votes, y= percent_dem)) +
  geom_abline() +
  geom_point() +
  labs(x = "Population of Hispanic or Latino (of any race)", y = "Percent of voters who voted Biden", title = "Population vs. Percent Biden Voters") +
  theme(text = element_text(size = 10),
        plot.title = element_text(size = 12))

## scatter plot for the relationship between total votes and percent_Dem
hisp_lat_log <- ggplot(train, aes(x = log10(total_votes), y= percent_dem)) +
  geom_abline() +
  geom_point() +
  labs(x = "Population of Hispanic or Latino (of any race) - Log", y = "Percent of voters who voted Biden", title = "Log Population vs. Percent Biden Voters") +
  theme(text = element_text(size = 10),
        plot.title = element_text(size = 12))

grid.arrange(hisp_lat, hisp_lat_log, ncol = 2)
 

   
# scatter plot for relationship between income_per_cap_2020 and percent_dem
income_2020 <- ggplot(train, aes(x = income_per_cap_2020, y = percent_dem)) +
  geom_abline() +
  geom_point() +
  labs(x = "Income per capita for the county in 2020", y = "Percent of voters who voted Biden", title = "Income 2020 vs. Biden voters") +
  theme(text = element_text(size = 10),
        plot.title = element_text(size = 12))

# scatter plot for relationship between income_per_cap_2020 and percent_dem
income_2020_log <- ggplot(train, aes(x = log10(income_per_cap_2020), y = percent_dem)) +
  geom_abline() +
  geom_point() +
  labs(x = "Income per capita for the county in 2020 - log", y = "Percent of voters who voted Biden", title = "Log Income 2020 vs. Biden voters") +
  theme(text = element_text(size = 10),
        plot.title = element_text(size = 12))

# Combine above two
grid.arrange(income_2020, income_2020_log, ncol = 2)
 

   
# scatter plot for relationship between income_per_cap_2020 and percent_dem, grouped by area type
ggplot(train, aes(x = log(income_per_cap_2020), y = percent_dem)) +
  geom_point() +
  facet_wrap(~x2013_code) +
  labs(x = 'Income per capita for the county in 2020 (missing for some counties)', y = 'Percent voters who voted Biden', title = 'Income 2020 vs. Biden voters by urban/rural code') +
  theme(text = element_text(size = 10),
        plot.title = element_text(size = 12))
 

   
## create a plot based on different education group
education <- train %>%
  select(percent_dem, c01_007e:c01_013e)

education_tidy <- education %>%
  pivot_longer(c("c01_007e", "c01_008e", "c01_009e", "c01_010e", "c01_011e", "c01_012e", "c01_013e"), names_to = "education", values_to = "population")

education_levels <- ggplot(education_tidy, aes(x = population, y = percent_dem, color = education)) +
  geom_point() +
  labs(x = "Population (25 years and over)", 
       y = "Percent of voters who voted Biden",
       title = "Educational Levels vs. Percent Biden Voters") +
  scale_color_discrete(name = "Education Level",
                       labels = c("Less than 9th grade", "9th to 12th grade, no diploma", "High school graduate", "Some college, no degree", "Associate's degree", "Bachelor's degree", "Graduate or professional degree")) + 
  theme(text = element_text(size = 8),
        plot.title = element_text(size = 12))

# Log transform

education$c01_007e <- log(education$c01_007e)
education$c01_008e <- log(education$c01_008e)
education$c01_009e <- log(education$c01_009e)
education$c01_010e <- log(education$c01_010e)
education$c01_011e <- log(education$c01_011e)
education$c01_012e <- log(education$c01_012e)
education$c01_013e <- log(education$c01_013e)

education_tidy_log <- education %>%
  pivot_longer(c("c01_007e", "c01_008e", "c01_009e", "c01_010e", "c01_011e", "c01_012e", "c01_013e"), names_to = "education", values_to = "population")

education_levels_log <- ggplot(education_tidy_log, aes(x = population, y = percent_dem, color = education)) +
  geom_point() +
  labs(x = "Population (25 years and over) - Log", 
       y = "Percent of voters who voted Biden",
       title = "Educational Levels (Log population) vs. Percent Biden Voters") +
  scale_color_discrete(name = "Education Level",
                       labels = c("Less than 9th grade", "9th to 12th grade, no diploma", "High school graduate", "Some college, no degree", "Associate's degree", "Bachelor's degree", "Graduate or professional degree")) + 
  theme(text = element_text(size = 8),
        plot.title = element_text(size = 12))

# Combine above two
grid.arrange(education_levels, education_levels_log, ncol = 1)
 

   
# Sample of similar groups and opposing groups

lower_ed <- ggplot(train, aes(x = log(c01_007e), y = log(c01_008e))) +
  geom_abline() +
  geom_point() +
  labs(x = "Less than 9th grade", y = "9th to 12th grade, no diploma", title = "Lower education group association - Log") +
  theme(text = element_text(size = 8))

higher_ed <- ggplot(train, aes(x = log(c01_012e), y = log(c01_013e))) +
  geom_abline() +
  geom_point() +
  labs(x = "Bachelor's degree", y = "Graduate or psrofessional degree", title = "Higher education group association - Log") +
  theme(text = element_text(size = 8))

low_high_ed <- ggplot(train, aes(x = log(c01_007e), y = log(c01_013e))) +
  geom_abline() +
  geom_point() +
  labs(x = "Less than 9th grade", y = "Graduate or professional degree", title = "Less than 9th vs Graduate or Prof - Log") +
  theme(text = element_text(size = 8))

# Combine above three
grid.arrange(lower_ed, higher_ed, low_high_ed, ncol = 2, nrow = 2)
 

   
## create a plot based on GDP by year
gdp <- train %>%
  select(percent_dem, gdp_2016:gdp_2020)

gdp_tidy <- gdp %>%
  pivot_longer(cols = c("gdp_2016","gdp_2017","gdp_2018","gdp_2019","gdp_2020"), names_to = "Year", values_to = "GDP")

gdp_plot <- ggplot(gdp_tidy, aes(x = GDP, y = percent_dem, color = Year)) +
  geom_point() +
  labs(x = "GDP for county", 
       y = "Percent of voters who voted Biden",
       title = "GDP for county vs. Percent Biden Voters") +
  scale_color_discrete(name = "Year",
                       labels = c("2016", "2017", "2018", "2019", "2020")) + 
  theme(text = element_text(size = 8),
        plot.title = element_text(size = 12))

# Log transform

gdp$gdp_2016 <- log(gdp$gdp_2016)
gdp$gdp_2017 <- log(gdp$gdp_2017)
gdp$gdp_2018 <- log(gdp$gdp_2018)
gdp$gdp_2019 <- log(gdp$gdp_2019)
gdp$gdp_2020 <- log(gdp$gdp_2020)

gdp_tidy_log <- gdp %>%
  pivot_longer(cols = c("gdp_2016","gdp_2017","gdp_2018","gdp_2019","gdp_2020"), names_to = "Year", values_to = "GDP")

gdp_plot_log <- ggplot(gdp_tidy_log, aes(x = GDP, y = percent_dem, color = Year)) +
  geom_point() +
  labs(x = "GDP for county - Log", 
       y = "Percent of voters who voted Biden",
       title = "Log GDP for county vs. Percent Biden Voters") +
  scale_color_discrete(name = "Year",
                       labels = c("2016", "2017", "2018", "2019", "2020")) + 
  theme(text = element_text(size = 8),
        plot.title = element_text(size = 12))

# Combine above two
grid.arrange(gdp_plot, gdp_plot_log, ncol = 1)
 

# Preprocessing / Recipes

   
# Remove non-numeric column ("id") and handle missing values
numeric_vars <- train[, !names(train) %in% c("name", "id")]
numeric_vars_na_omit <- na.omit(numeric_vars)  # Remove rows with missing values

# Calculate the correlation matrix for numeric variables
correlation_matrix <- cor(numeric_vars_na_omit)

# Identify highly correlated variables
highly_correlated <- findCorrelation(correlation_matrix, cutoff = 0.75)

# Calculate skewness for each column in the dataframe df
skew_values <- sapply(numeric_vars, skewness)

# Set a threshold for skewness
threshold <- 1.0

# Get column names with high skewness
high_skew_columns <- names(skew_values[abs(skew_values) > threshold])
high_skew_columns <- as.character(high_skew_columns)
high_skew_columns <- na.omit(high_skew_columns)

education_cols = c("c01_003e" , "c01_004e", "c01_005e","c01_006e" , "c01_007e", "c01_008e"  ,"c01_009e" , "c01_010e","c01_011e" ,"c01_012e" ,"c01_013e" , "c01_014e","c01_015e", "c01_016e", "c01_017e",                           "c01_018e" ,"c01_019e", "c01_020e" , "c01_021e", "c01_022e", "c01_023e", "c01_024e", "c01_025e", "c01_026e", "c01_027e")

cols_to_drop_1 = c("x0001e", "x0018e","x0019e", "x0020e", "x0021e", "x0022e", "x0023e","x0024e" , "x0025e", "x0026e", "x0027e", "x0029e", "x0030e", "x0031e",  "x0034e", "x0058e", "x0062e", "x0064e", "x0065e",   "x0066e", "x0067e", "x0068e", "x0069e", "x0076e" , "x0077e", "x0078e", "x0079e","x0080e","x0081e","x0082e","x0083e", "x0002e","x0003e","x0005e","x0006e","x0007e","x0008e","x0009e","x0010e","x0011e","x0012e","x0013e","x0014e","x0015e", "x0016e", "x0017e")


# Remove "name" column from train
train <- subset(train, select = -name)
 

   
# Create all recipes

basic_recipe <- 
  recipe(percent_dem ~ ., data = train) %>%
  step_rm(id) %>%
  step_log(all_of(high_skew_columns), base = 10, offset = 0.001) %>%
  step_impute_knn(all_predictors())

# Add normalization step
normalized_recipe <-
  basic_recipe %>%
  step_normalize(all_numeric_predictors())

# Add interaction columns
interaction_recipe <-
  normalized_recipe %>%
  step_interact(terms = ~ income_per_cap_2016:x2013_code +
                  income_per_cap_2017:x2013_code +
                  income_per_cap_2018:x2013_code +
                  income_per_cap_2019:x2013_code + 
                  income_per_cap_2020:x2013_code)

# Account for correlation/collinearity and zero/low variance columns
cor_var_recipe <-
  interaction_recipe %>%
  step_corr(all_predictors(), threshold = 0.75) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

# Create custom recipe with multiple new columns calculating percentages of different groups based on total population

code2013 <- c("one", "two", "three", "four", "five", "six")

custom_recipe <- recipe(percent_dem ~ ., data = train) %>%
  step_rm(id) %>%
  step_impute_knn(all_predictors()) %>%
  step_mutate(pct_male = x0002e/x0001e) %>%
  step_mutate(pct_female = x0003e/x0001e)  %>%
  step_mutate(pct_male_18over = x0026e / x0001e) %>%
  step_mutate(pct_female_18over = x0027e / x0001e) %>%
  step_mutate(pct_21andover = x0022e/x0001e) %>%
  step_mutate(pct_62andove = x0023e/x0001e) %>%
  step_mutate(pct_female_65over = x0031e / x0001e) %>%
  step_mutate(pct_male_65over = x0030e / x0001e) %>%
  step_mutate(pct_16over = x0020e/ x0001e)%>%
  step_mutate(pct_18over = x0025e/ x0001e)%>%
  step_mutate(pct_0to19 = (x0005e + x0006e + x0007e + x0008e) / x0001e) %>%
  step_mutate(pct_20to34 = (x0009e + x0010e) / x0001e) %>%
  step_mutate(pct_35to54 = (x0011e + x0012e) / x0001e) %>%
  step_mutate(pct_55to64 = (x0013e + x0014e) / x0001e) %>%
  step_mutate(pct_65andovr = (x0015e + x0016e + x0017e) / x0001e) %>%
  step_mutate(pct_18o_citizen_m = x0088e / x0001e) %>%
  step_mutate(pct_18o_citizen_f = x0089e / x0001e) %>%
  step_mutate(pct_18o_citizen = x0087e / x0001e) %>%
  step_mutate(pct_onerace = x0034e / x0001e)  %>%
  step_mutate(pct_morerace = x0035e / x0001e) %>%
  step_mutate(pct_white = x0037e / x0001e) %>%
  step_mutate(pct_black = x0038e / x0001e) %>%
  step_mutate(pct_indian = x0039e / x0001e) %>%
  step_mutate(pct_asian = x0044e / x0001e) %>%
  step_mutate(pct_api = x0052e / x0001e) %>%
  step_mutate(pct_otherrace = x0057e / x0001e) %>%
  step_mutate(pct_white_c = x0064e / x0001e)  %>%
  step_mutate(pct_black_c = x0065e / x0001e)  %>%
  step_mutate(pct_indian_c = x0066e / x0001e)  %>%
  step_mutate(pct_asian_c = x0067e / x0001e)  %>%
  step_mutate(pct_api_c = x0068e / x0001e)  %>%
  step_mutate(pct_other_c = x0069e / x0001e)  %>%
  step_mutate(pct_his = x0071e / x0001e) %>%
  step_mutate(pct_cherokee = x0040e/x0001e) %>%
  step_mutate(pct_chippewa = x0041e/x0001e) %>%
  step_mutate(pct_navajo = x0042e/x0001e) %>%
  step_mutate(pct_sioux = x0043e/x0001e) %>%
  step_mutate(pct_indian = x0045e/x0001e) %>%
  step_mutate(pct_chinese = x0046e/x0001e) %>%
  step_mutate(pct_filipino = x0047e/x0001e) %>%
  step_mutate(pct_japanese = x0048e/x0001e) %>%
  step_mutate(pct_korean = x0049e/x0001e) %>%
  step_mutate(pct_viet = x0050e/x0001e) %>%
  step_mutate(pct_othera = x0051e/x0001e) %>%
  step_mutate(pct_hawaii = x0053e/x0001e) %>%
  step_mutate(pct_chamorro = x0054e/x0001e) %>%
  step_mutate(pct_samoan = x0055e/x0001e) %>%
  step_mutate(pct_otherpi = x0056e/x0001e) %>%
  step_mutate(pct_mexican = x0072e/x0001e) %>%
  step_mutate(pct_puertorican = x0073e/x0001e) %>%
  step_mutate(pct_cuban = x0074e/x0001e) %>%
  step_mutate(pct_otherh = x0075e/x0001e) %>%
  step_mutate(pct_whiteandblack = x0059e/x0001e) %>%
  step_mutate(pct_whiteandindian = x0060e/x0001e) %>%
  step_mutate(pct_whiteandasian = x0061e/x0001e) %>%
  step_mutate(pct_indianandblack = x0062e/x0001e) %>%
  step_mutate(pc_nothispanic_w = x0077e / x0001e) %>%
  step_mutate(pc_nothispanic_b = x0078e / x0001e) %>%
  step_mutate(pc_nothispanic_i = x0079e / x0001e) %>%
  step_mutate(pc_nothispanic_a = x0080e / x0001e) %>%
  step_mutate(pc_nothispanic_hpi = x0081e / x0001e) %>%
  step_mutate(pc_nothispanic_o = x0082e / x0001e) %>%
  step_mutate(pc_nothispanic_more = x0083e / x0001e) %>%
  step_mutate(pc_nothispanic_more_o = x0084e / x0001e) %>%
  step_mutate(pc_nothispanic_three = x0085e / x0001e) %>%
  step_mutate(pct_lesshighschool_18to24 = c01_002e / ifelse(c01_001e != 0, c01_001e, 1))  %>%
  step_mutate(pct_highschool_18to24 = c01_003e/ifelse(c01_001e != 0, c01_001e, 1)) %>%
  step_mutate(pc_college_18to24 = c01_004e/ifelse(c01_001e != 0, c01_001e, 1)) %>%
  step_mutate(pc_bachelor_18to24 = c01_005e/ifelse(c01_001e != 0, c01_001e, 1)) %>%
  step_mutate(pc_lesshighschool_25over = (c01_007e + c01_008e)/c01_006e) %>%
  step_mutate(pc_highschool_25over = c01_009e/c01_006e) %>%
  step_mutate(pc_college_25over = (c01_010e + c01_011e)/c01_006e) %>%
  step_mutate(pc_bachelorover_25over = c01_015e/c01_006e) %>%
  step_mutate(pc_highschoolover_25over = c01_014e/c01_006e) %>%
  step_mutate(pc_associate_25over = c01_011e / c01_006e ) %>%
  step_mutate(pc_bachlor_25over = c01_012e / c01_006e ) %>%
  step_mutate(pc_graduate_25over = c01_013e / c01_006e ) %>%
  step_mutate(pc_highschool_25to34 = c01_017e/ c01_016e) %>%
  step_mutate(pc_bachelor_25to34 = c01_018e/ c01_016e) %>%
  step_mutate(pc_highschool_35to44 = c01_020e/ c01_019e) %>%
  step_mutate(pc_bachelor_35to44 = c01_021e/ c01_019e) %>%
  step_mutate(pc_highschool_45to64 = c01_023e/ c01_022e) %>%
  step_mutate(pc_bachelor_45to64 = c01_024e/ c01_022e) %>%
  step_mutate(pc_highschool_65over = c01_026e/ c01_025e) %>%
  step_mutate(pc_bachelor_65over = c01_027e/ c01_025e) %>%
  step_mutate(mean_income_per_cap = (income_per_cap_2016 + income_per_cap_2017+ income_per_cap_2018 + income_per_cap_2019+ income_per_cap_2020) / 5)  %>%
  step_mutate(mean_gdp = (gdp_2016+ gdp_2017+ gdp_2018 + gdp_2019+ gdp_2020)/5) %>%
  step_mutate(pc_vote_population = total_votes / x0001e) %>%
  step_mutate(pc_vote_eligible = total_votes/x0087e) %>%
  step_rm(x0002e, x0009e, x0010e, x0003e, x0005e, x0006e, x0007e, x0008e, x0033e, x0011e, x0012e, x0013e, x0014e, x0015e, x0016e, x0017e) %>%
  step_rm(x0020e, x0021e, x0022e, x0023e, x0024e, x0025e, x0029e, x0087e, x0088e, x0089e) %>% #age
  step_rm(x0034e, x0035e, x0036e, x0037e, x0038e, x0039e, x0040e, x0052e, x0044e,x0057e, x0058e, x0064e, x0065e,x0066e, x0067e, x0068e, x0069e, x0071e) %>% #race
  step_rm(c01_002e, c01_015e,c01_011e,c01_010e, c01_009e, c01_008e, c01_007e,  c01_005e, c01_004e, c01_003e) %>% 
  step_rm(c01_017e, c01_018e, c01_019e, c01_020e, c01_021e, c01_022e, c01_023e, c01_024e, c01_025e, c01_026e, c01_027e)%>% 
  step_rm(x0041e, x0042e, x0043e, x0045e, x0046e, x0047e, x0048e, x0049e, x0050e, x0051e, x0053e, x0054e, x0055e, x0056e) %>% 
  step_rm(matches("^x007[2-9]e$|^x008[0-5]e$"))  %>%
  step_rm(x0019e, x0026e, x0027e, x0031e, x0030e, x0060e, x0061e, x0062e, x0059e) %>%
  step_rm(c01_001e, c01_006e, c01_012e, c01_013e, c01_016e)%>%
  step_rm(c01_014e)%>%
  step_rm(total_votes) %>%
  step_num2factor(x2013_code, levels = code2013) %>%
  step_dummy(x2013_code) %>%
  step_zv(all_predictors())
 

# Candidate models / Model evaluation / tuning

# 1.  Linear Regression (baseline testing for recipes)

   
# Initialize candidate models for testing

# Linear Regression (baseline testing for selecting best recipe)

linear_regression_spec <- 
  linear_reg() %>%
  set_engine("lm")
 

# 2.  Random forest

   
# Random forest (basic)

random_forest_spec <-
  rand_forest(trees = 100) %>% 
  set_mode("regression") %>% 
  set_engine("ranger", importance = "impurity")
 

# 3.  K-nearest neighbors

   
# K-nearest neighbors

knn_spec <-
  nearest_neighbor(
    mode = "regression", 
    neighbors = tune("k")
  ) %>%
  set_engine("kknn")
 

# 4.  Support vector machine

   
# Support vector machine

svm_spec <- 
  svm_rbf(
    cost = tune("cost"), 
    rbf_sigma = tune("sigma")
  ) %>%
  set_engine("kernlab") %>%
  set_mode("regression")
 

# 5.  Boosted trees - xgboost engine

   
# Boosted trees - xgboost engine

bt_xgboost_spec <- 
  boost_tree(
    mode = "regression", 
    trees = 100,
    learn_rate = 0.05,
    # Define a tuning grid
    tree_depth = tune("tree_depth"),
    mtry = tune("mtry"),
    min_n = tune("min_n"),
    loss_reduction = tune("loss_red")
  ) %>%
  set_engine("xgboost")
 

# 6.  Boosted trees - lightgbm engine

   
# Boosted trees - lightgbm engine

bt_lightgbm_spec <- 
  boost_tree(
    mode = "regression", 
    trees = 100,
    learn_rate = 0.05,
    # Define a tuning grid
    tree_depth = tune("tree_depth"),
    mtry = tune("mtry"),
    min_n = tune("min_n"),
    loss_reduction = tune("loss_red")
  ) %>% 
  set_engine("lightgbm")
 

# Workflow sets (recipes x models)

   
# Run this part to select ideal recipe (baseline)

# Set up preprocess object containing all candidate recipes
preproc <- list(basic = basic_recipe,
                normal = normalized_recipe,
                interact = interaction_recipe,
                cor_var = cor_var_recipe,
                cust = custom_recipe)

keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)
 

   

# Create a workflow_set to test all candidate recipes against baseline linear regression model
lm_models <- workflow_set(preproc = preproc,
                          models = list(lm =  linear_regression_spec),
                          cross = FALSE)

# Use workflow_map to call in v-fold cross validation data
lm_models <- 
  lm_models %>%
  workflow_map("fit_resamples",
               # Options to `workflow_map()`:
               seed = 1, verbose = TRUE,
               # Options to `fit_resamples()`:
               resamples = train_folds, control = keep_pred)
 

   
# This shows that the custom recipe performs best

lm_models %>%
  collect_metrics() %>%
  filter(.metric == "rmse")
 

   
autoplot(lm_models)
 

   
# Random forest individual fit

rf_workflow <- 
  workflow() %>% 
  add_recipe(custom_recipe) %>% 
  add_model(random_forest_spec)

rf_res <- 
  rf_workflow %>%
  fit_resamples(resamples = train_folds,
                control = control_resamples(save_pred = TRUE))
 

   
rf_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse")
 

   
# KNN individual fit

knn_wflow <- 
  workflow() %>% 
  add_model(knn_spec) %>%
  add_recipe(custom_recipe)

knn_res <- 
  tune_grid(
    knn_wflow,
    resamples = train_folds,
    metrics = metric,
    grid = 4,
    control = control_stack_grid()
  )
 

   
show_best(knn_res, metric = "rmse")
 

   
autoplot(knn_res)
 

   
# SVM individual fit

svm_wflow <- 
  workflow() %>% 
  add_model(svm_spec) %>%
  add_recipe(custom_recipe)

svm_res <- 
  tune_grid(
    svm_wflow, 
    resamples = train_folds, 
    grid = 6,
    metrics = metric,
    control = control_stack_grid()
  )
 

   
show_best(svm_res, metric = "rmse")
 

   
autoplot(svm_res)
 

   
# Boosted trees - xgboost fit

bt_xgboost_workflow <- workflow() %>% 
  add_model(bt_xgboost_spec) %>% 
  add_recipe(custom_recipe)

# # Define a tuning grid
# grid_xgboost <- grid_regular(
#   tree_depth(range = c(1, 30), trans = NULL),  # Adjust the range based on your knowledge of the problem
#   mtry(range = c(2, 20), trans = NULL),  # Adjust the range based on your knowledge of the problem
#   min_n(range = c(5, 50), trans = NULL),  # Adjust the range based on your knowledge of the problem
#   loss_reduction(range = c(0, 0.1), trans = NULL),  # Adjust the range based on your knowledge of the problem
#   levels = 5  # or another number to control the grid size
# )

bt_xgboost_res <- 
  tune_grid(
    bt_xgboost_workflow,
    resamples = train_folds,
    grid = 5,
    metrics = metric,
    control = control_stack_grid()
  )
 

   
show_best(bt_xgboost_res, metric = "rmse")
 

   
autoplot(bt_xgboost_res)
 

   
# Boosted trees - lightgbm fit

bt_light_workflow <- workflow() %>% 
  add_model(bt_lightgbm_spec) %>% 
  add_recipe(custom_recipe)

bt_light_res <- 
  tune_grid(
    bt_light_workflow,
    resamples = train_folds,
    grid = 5,
    metrics = metric,
    control = control_stack_grid()
  )
 

   
show_best(bt_light_res, metric = "rmse")
 

   
autoplot(bt_light_res)
 

   
# Stacked model

# Define stack object adding the top four candidate models
test_stack <- 
  stacks() %>%
  add_candidates(knn_res) %>%
  add_candidates(svm_res) %>%
  add_candidates(bt_xgboost_res) %>%
  add_candidates(bt_light_res)

test_stack <-
  test_stack %>%
  blend_predictions()
 

   
autoplot(test_stack)
 

   
autoplot(test_stack, type = "weights")
 

   

# Fit the models based on evaluation of the stack function
test_stack <-
  test_stack %>%
  fit_members()
 

   

# Generate final predictions

test_res_stacked <- 
  test_stack %>% 
  predict(test) %>%
  cbind(test %>% select(id)) %>%
  select(id, .pred)

test_res_stacked <- 
  test_res_stacked %>% 
  rename(percent_dem = .pred) %>% 
  write_csv("stacked_predictions.csv")

print(head(test_res_stacked, n = 15))
 