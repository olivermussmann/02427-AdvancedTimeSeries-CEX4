clc; clear; close all;

%% ------------------------- Collect data ---------------------------------

% Read the CSV file
X = readtable('cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
X = rmmissing(X); % Remove rows containing NaN's

% Extract relevant columns
windPower = X.p;   % Measured wind power
windSpeed = X.Ws1; % 1-hour forecasted wind speed (input)
windSpeed2 = X.Ws2; % 2-hour forecasted wind speed (input)
windSpeed3 = X.Ws3; % 3-hour forecasted wind speed (input)


% Ensure valid data for fitting (exclude zeros or negatives)
validIdx = windPower > 0 & windSpeed > 0;
windPower = windPower(validIdx);
windSpeed = windSpeed(validIdx);
windSpeed2 = windSpeed2(validIdx);
windSpeed3 = windSpeed3(validIdx);

% Number of observations
n = length(windPower);

% Define the split index
splitIndex = n - 1000;

% Split into training and testing sets
trainWindPower = windPower(1:splitIndex);
trainWindSpeed = windSpeed(1:splitIndex);

testWindPower = windPower(splitIndex + 1:end);
testWindSpeed = windSpeed(splitIndex + 1:end);
testWindSpeed2 = windSpeed2(splitIndex + 1:end);
testWindSpeed3 = windSpeed3(splitIndex + 1:end);

%% ------------------- Sigmoid Power Curve Regression --------------------

% Sigmoid function definition
sigmoidFunc = @(params, w) params(1) ./ (1 + exp(-params(2) * (w - params(3))));

% Initial guesses for parameters: [Pmax, a, b]
initialParams = [max(trainWindPower), 0.1, mean(trainWindSpeed)];

% Objective function for fitting
objectiveFunc = @(params) sum((trainWindPower - sigmoidFunc(params, trainWindSpeed)).^2);

% Optimize parameters using non-linear least squares
optimalParams = fminsearch(objectiveFunc, initialParams);

% Extract optimized parameters
Pmax = optimalParams(1); % Maximum power
a = optimalParams(2);    % Steepness of the curve
b = optimalParams(3);    % Inflection point (wind speed at half-max power)

disp('Optimized Sigmoid Parameters:');
disp(['Pmax: ', num2str(Pmax)]);
disp(['a: ', num2str(a)]);
disp(['b: ', num2str(b)]);

%% --------------------- Power Curve Prediction --------------------------


% Predict for 1-step, 2-step, and 3-step
steps = 1:3;
nSteps = length(steps);
y_pred_sigmoid = zeros(length(testWindPower), nSteps);

for t = 1:length(testWindPower)
    % 1-step prediction
    y_pred_sigmoid(t, 1) = sigmoidFunc(optimalParams, testWindSpeed(t));
    if y_pred_sigmoid(t, 1) < 0
        y_pred_sigmoid(t, 1) = 0;
    end

    % Multi-step predictions
    for step = 2:nSteps
        if t + step - 1 <= length(testWindPower)
            % Select the correct wind speed input
            if step == 2
                wind_speed = testWindSpeed2(t + step - 1);
            else
                wind_speed = testWindSpeed3(t + step - 1);
            end
            
            % Predict using the sigmoid function
            y_pred_sigmoid(t + step - 1, step) = sigmoidFunc(optimalParams, wind_speed);
            
            % Ensure non-negative predictions
            y_pred_sigmoid(t + step - 1, step) = max(y_pred_sigmoid(t + step - 1, step), 0);
        end
    end
end

%% ------------------------- Performance Metrics -------------------------

% Residuals and metrics
residuals_sigmoid = testWindPower - y_pred_sigmoid;
RSS_sigmoid = sum(residuals_sigmoid.^2, 1);
RMSE_sigmoid = sqrt(mean(residuals_sigmoid.^2, 1));
params = 3; % Number of parameters (Pmax, a, b)
AIC_sigmoid = length(testWindPower) * log(RSS_sigmoid / length(testWindPower)) + 2 * params;
BIC_sigmoid = length(testWindPower) * log(RSS_sigmoid / length(testWindPower)) + params * log(length(testWindPower));

% Display metrics for each step
for step = steps
    disp(['Metrics for ', num2str(step), '-step predictions:']);
    disp(['RMSE: ', num2str(RMSE_sigmoid(step))]);
    disp(['AIC: ', num2str(AIC_sigmoid(step))]);
    disp(['BIC: ', num2str(BIC_sigmoid(step))]);
end

%% ------------------------- Plot Power Curve ----------------------------

% Plot estimated sigmoid power curve
figure;
hold on;
x = linspace(min(trainWindSpeed), max(trainWindSpeed), 100)';
y = sigmoidFunc(optimalParams, x);
scatter(trainWindSpeed, trainWindPower, 'k.', 'DisplayName', 'Training Data');
plot(x, y, 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated Power Curve');
xlabel('Wind Speed', 'Interpreter', 'latex');
ylabel('Measured Wind Power', 'Interpreter', 'latex');
xlim([0 35])
ylim([0 20])
title('Estimated Sigmoid Power Curve', 'Interpreter', 'latex');
legend('Location', 'southeast', 'Interpreter', 'latex');
set(gca, 'FontSize', 13, 'TickLabelInterpreter', 'latex')
grid on;
box on;
hold off;

%% ------------------------- Plot Multi-Step Predictions -----------------

% Titles for prediction plots
titles = {'Power curve - 1-step Predictions', 'Power curve - 2-step Predictions', 'Power curve - 3-step Predictions'};

% Plot predictions
time = (1:length(testWindPower))';
f = figure('Units', 'pixels', 'Position', [600, 300, 800, 600]);
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:3
    ax = nexttile;
    hold on;
    plot(time(end-100:end), testWindPower(end-100:end), 'k', 'LineWidth', 2, 'DisplayName', 'Actual');
    plot(time(end-100:end), y_pred_sigmoid(end-100:end, i), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
    title(titles{i}, 'Interpreter', 'latex');
    xlabel('Time Index', 'Interpreter','latex');
    ylabel('Wind Power', 'Interpreter','latex');
    if i == 1
        legend('Location', 'northwest', 'Interpreter','latex');
    end
    set(ax, 'FontSize', 13, 'TickLabelInterpreter', 'latex')
    grid on;
    hold off;
end

%% Save script to Github

current_file = 'PowerCurve.m';
new_directory = '/home/olivermussmann/Documents/GitHub/02427-AdvancedTimeSeries-CEX4/matlab_code';
copyfile(current_file, new_directory);

