clc; clear; close all;

%% ------------------------- Collect Data ---------------------------------

% Read the CSV file
X = readtable('comp_ex_4_scripts_2011/data/cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
X = rmmissing(X); % Remove rows containing NaN's
X.t = hours(X.t - X.t(1));

% Extract relevant columns
windPower = X.p;   % Measured wind power
windSpeed = X.Ws1; % 1-hour forecasted wind speed (exogenous input)
windSpeed2 = X.Ws2; % 2-hour forecasted wind speed (exogenous input)
windSpeed3 = X.Ws3; % 3-hour forecasted wind speed (exogenous input)
windDir = X.Wd1;   % 1-hour forecasted wind direction (for regimes)
windDir2 = X.Wd2;   % 2-hour forecasted wind direction (for regimes)
windDir3 = X.Wd3;   % 3-hour forecasted wind direction (for regimes)

% Number of observations
n = length(windPower);

% Define the split index
splitIndex = n - 2000;

% Split into training and testing sets
trainWindPower = windPower(1:splitIndex);
trainWindSpeed = windSpeed(1:splitIndex);
trainWindDir = windDir(1:splitIndex);

testWindPower = windPower(splitIndex + 1:end);

testWindSpeed = windSpeed(splitIndex + 1:end);
testWindSpeed2 = windSpeed2(splitIndex + 1:end);
testWindSpeed3 = windSpeed3(splitIndex + 1:end);

testWindDir = windDir(splitIndex + 1:end);
testWindDir2 = windDir2(splitIndex + 1:end);
testWindDir3 = windDir3(splitIndex + 1:end);

testTime = X.t(splitIndex + 1:end);

%% ------------------------- Regime Definitions --------------------------

% Define the regimes
R1 = @(dir) (dir >= 300 | dir < 60);   % 300°-360° and 0°-60°
R2 = @(dir) (dir >= 60 & dir < 180);  % 60°-180°
R3 = @(dir) (dir >= 180 & dir < 300); % 180°-300°

% Identify regimes for training data
regimeTrain = zeros(size(trainWindDir));
regimeTrain(R1(trainWindDir)) = 1;
regimeTrain(R2(trainWindDir)) = 2;
regimeTrain(R3(trainWindDir)) = 3;

% Identify regimes for testing data
regimeTest = zeros(size(testWindDir));
regimeTest(R1(testWindDir)) = 1;
regimeTest(R2(testWindDir)) = 2;
regimeTest(R3(testWindDir)) = 3;

regimeTest2 = zeros(size(testWindDir2));
regimeTest2(R1(testWindDir2)) = 1;
regimeTest2(R2(testWindDir2)) = 2;
regimeTest2(R3(testWindDir2)) = 3;

regimeTest3 = zeros(size(testWindDir3));
regimeTest3(R1(testWindDir3)) = 1;
regimeTest3(R2(testWindDir3)) = 2;
regimeTest3(R3(testWindDir3)) = 3;

%% ------------------------- TARX Model Training -------------------------

% AR lag order
lagOrder = 7;
exoOrder = 5;

% Prepare data storage for models
models_TARX = cell(1, 3);

% Train TARX model for each regime
for regime = 1:3
    % Filter data for the current regime
    idx = (regimeTrain == regime);
    y_train = trainWindPower(idx);
    x_train = trainWindSpeed(idx);
    nTrain = length(y_train);
    
    % Construct lagged AR and exogenous terms
    if nTrain > lagOrder
        X_AR = zeros(nTrain - lagOrder, lagOrder);  % AR lags
        X_X = zeros(nTrain - lagOrder, exoOrder);  % Exogenous terms
        
        for i = 1:lagOrder
            X_AR(:, i) = y_train(lagOrder + 1 - i:nTrain - i);
        end

        for i = 1:exoOrder
            X_X(:, i) = x_train(lagOrder + 1 - i:nTrain - i);
        end

        % Combine AR and exogenous terms
        X_combined = [X_AR, X_X];
        y_target = y_train(lagOrder + 1:end);
        
        % Fit linear model for the current regime
        models_TARX{regime} = fitlm(X_combined, y_target);
    end
end

%% ------------------------- TARX Multi-Step Predictions -----------------

% Number of observations
n = length(windPower);

% Combine training and test data for recursive predictions
Power_data = [trainWindPower; testWindPower];

% Predict for test data (one-step, two-step, three-step ahead)
y_pred_TARX = zeros(length(testWindPower), 3);
for t = splitIndex + 1:n
    % One-step prediction
    currentRegime = regimeTest(t - splitIndex);

    mdl = models_TARX{currentRegime};

    X_AR = flip(Power_data(t-lagOrder:t-1))';
    X_X = flip(windSpeed(t-exoOrder:t-1))';
    X_combined = [X_AR, X_X];
    
    y_pred_TARX(t - splitIndex, 1) = predict(mdl, X_combined);
    y_pred_TARX(t - splitIndex, 1) = max(y_pred_TARX(t - splitIndex, 1), 0); % Ensure non-negative

    % Two-step prediction
    if t - splitIndex > 1
        currentRegime2 = regimeTest2(t - splitIndex);

        mdl2 = models_TARX{currentRegime2};

        X_AR_2 = [y_pred_TARX(t - splitIndex - 1, 1), flip(Power_data(t-lagOrder:t-2))'];
        X_X_2 = [windSpeed2(t-1), flip(windSpeed(t-exoOrder:t-2))'];
        X_combined2 = [X_AR_2, X_X_2];
        
        y_pred_TARX(t - splitIndex, 2) = predict(mdl2, X_combined2);
        y_pred_TARX(t - splitIndex, 2) = max(y_pred_TARX(t - splitIndex, 2), 0);
    end

    % Three-step prediction
    if t - splitIndex > 2
        currentRegime3 = regimeTest3(t - splitIndex);
        mdl3 = models_TARX{currentRegime3};
        X_AR_3 = [y_pred_TARX(t - splitIndex - 1, 2), y_pred_TARX(t - splitIndex - 2, 1), flip(Power_data(t-lagOrder:t-3))'];
        X_X_3 = [windSpeed3(t-1), windSpeed2(t-2), flip(windSpeed(t-exoOrder:t-3))'];
        X_combined3 = [X_AR_3, X_X_3];
        y_pred_TARX(t - splitIndex, 3) = predict(mdl3, X_combined3);
        y_pred_TARX(t - splitIndex, 3) = max(y_pred_TARX(t - splitIndex, 3), 0);
    end
end

%% ------------------------- Metrics and Plotting ------------------------

% Define data used for evaluation of predictions
y_pred_eval = y_pred_TARX(end-1000+1:end, :);
testWindPower_eval = testWindPower(end-1000+1:end);


% Calculate and print metrics
[RMSE, AIC, BIC] = calculateMetrics(testWindPower_eval, y_pred_eval, mdl.NumEstimatedCoefficients);

fprintf('Prediction Metrics:\n');
for i = 1:3
    fprintf('Step-%d Predictions:\n', i);
    fprintf('  RMSE: %.4f\n', RMSE(i));
    fprintf('  AIC: %.4f\n', AIC(i));
    fprintf('  BIC: %.4f\n', BIC(i));
end

% Plot predictions
titles = {'TARX(7,5) - 1-step Predictions', 'TARX(7,5) - 2-step Predictions', 'TARX(7,5) - 3-step Predictions'};
f = plotPredictions(testWindPower, y_pred_TARX, testTime(end-100:end), titles);

function [RMSE, AIC, BIC] = calculateMetrics(y_true, y_pred, k)
    RMSE = zeros(1, 3);
    AIC = zeros(1, 3);
    BIC = zeros(1, 3);

    for i = 1:3
        % Valid indices for comparison
        validIdx = ~isnan(y_pred(:, i));
        y_true_valid = y_true(validIdx); % Where i is the step-ahead (1, 2, or 3)
        y_pred_valid = y_pred(validIdx, i);
        
        % Residuals
        residuals = y_true_valid - y_pred_valid;

        % Residual Sum of Squares
        RSS = sum(residuals.^2);
        N = length(y_true_valid);

        % Compute Metrics
        RMSE(i) = sqrt(RSS / N);
        AIC(i) = N * log(RSS / N) + 2 * k;
        BIC(i) = N * log(RSS / N) + k * log(N);
    end
end

function f = plotPredictions(y_true, y_pred, time, titleName)
    f = figure('Units', 'pixels', 'Position', [600, 300, 800, 600]);
    tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:3
        ax = nexttile;
        hold on;

        plot_length = length(time);
        time_valid = time(i:end);
        y_true_valid = y_true(end - plot_length:end-i);
        y_pred_valid = y_pred(end - plot_length+i:end, i);
        
        disp(length(time_valid))
        disp(length(y_true_valid))
        disp(length(y_pred_valid))

        plot(time_valid, y_true_valid, 'k', 'DisplayName', 'Actual', 'LineWidth', 2);
        plot(time_valid, y_pred_valid, 'r--', 'DisplayName', 'Predicted', 'LineWidth', 1.5);
        
        title(titleName{i}, 'Interpreter', 'latex');
        xlabel('Time', 'Interpreter', 'latex');
        ylabel('Wind Power', 'Interpreter', 'latex');
        legend('Location', 'northwest', 'Interpreter', 'latex');
        grid on;
        hold off;
    end
end

% Save plot as a PDF
exportgraphics(f, 'TARX_pred.pdf', 'ContentType', 'vector');

%% Save script to Github

current_file = 'TARX.m';
new_directory = '/home/olivermussmann/Documents/GitHub/02427-AdvancedTimeSeries-CEX4/matlab_code';
copyfile(current_file, new_directory);