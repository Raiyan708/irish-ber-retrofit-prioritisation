# Leakage Notes

The target variable in this project is created from the BER rating. Because of this, BER-related output variables should not be used as model predictors.

If these variables were included, the model could learn information that directly reveals the target instead of learning from independent dwelling characteristics.

For this reason, variables such as BER rating outputs, CO2 rating outputs, and calculated energy-performance outputs are removed from the modelling feature set.

This helps make the model evaluation more realistic and avoids overly optimistic results caused by target leakage.
