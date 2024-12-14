clc; clear; close all;


%% ------------------------- Collect data ---------------------------------

% The table, X, constists of the following data:

% t:        Time [h]
% toy:      Time of year [days]
% p:        Measured average wind power
% Ws1:      1-hour ahead forecasted wind speed
% Wd1:      1-hour ahead forecasted wind direction
% T1:       1-hour ahead forecasted temperature
% Ws2:      2-hour ahead forecasted wind speed
% Wd2:      2-hour ahead forecasted wind direction
% T2:       2-hour ahead forecasted temperature
% Ws3:      3-hour ahead forecasted wind speed
% Wd3:      3-hour ahead forecasted wind direction
% T3:       3-hour ahead forecasted temperature


% Read the CSV file
X = readtable('cex4WindDataInterpolated.csv', 'Delimiter', ',', 'ReadVariableNames', true);
X = rmmissing(X); % Remove rows containing NaN's

% Convert the 't' column to datetime format with UTC timezone
X.t = datetime(X.t, 'InputFormat', 'yyyy-MM-dd HH:mm:ss', 'TimeZone', 'UTC');

% Convert the 't' column to relative hours
X.t = hours(X.t - X.t(1));





%% -------------------------- Data exploration ----------------------------


% Basic Dataset Overview
disp('Dataset Dimensions:');
disp(size(X)); % Number of rows and columns
disp('First few rows of the dataset:');
disp(head(X)); % Display the first few rows of the table


% Visualize Wind Power Over Time
f = figure('Units', 'pixels', 'Position', [600, 300, 800, 500]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

data = {X.t, X.p, X.t(1:1001), X.p(1:1001)};
x_limits = [0, 37940, 0, 1000];
y_limits = [0, 1.1*max(data{2}), 0, 1.1*max(data{4})];

for i = 1:2
    ax = nexttile;
    plot(data{2*i-1}, data{2*i}, 'LineWidth', 1.5, 'Color',[39 155 72]/255);
    xlabel('Time [h]', 'Interpreter', 'latex');
    ylabel('Measured Wind Power', 'Interpreter', 'latex');
    set(ax, 'FontSize', 13, 'TickLabelInterpreter', 'latex')
    xlim([x_limits(2*i-1) x_limits(2*i)])
    ylim([y_limits(2*i-1) y_limits(2*i)])
    grid on;
    box on;
end
exportgraphics(f,'power_data.pdf', 'ContentType', 'vector')

%%

% Seasonal Trends in Wind Power
% Group data by day of year (toy) and calculate mean power
dailyPower = groupsummary(X, 'toy', 'mean', 'p');
f = figure('Units', 'pixels', 'Position', [600, 300, 800, 500]);
plot(dailyPower.toy, dailyPower.mean_p, 'LineWidth', 1.5, 'Color', [39 155 72]/255);
xlabel('Day of Year', 'Interpreter','latex');
ylabel('Average Wind Power', 'Interpreter','latex');
title('Seasonal Trends in Wind Power', 'Interpreter','latex');
set(gca, 'FontSize', 13, 'TickLabelInterpreter', 'latex')
xlim([0 dailyPower.toy(end)])
grid on;
box on;
exportgraphics(f,'seasonality_plot.pdf', 'ContentType', 'vector')

%%

% Correlation Matrix for Key Variables
disp('Correlation Matrix:');
selectedVars = X{:, {'p', 'Ws1', 'Wd1', 'T1', 'toy'}};
correlationMatrix = corr(selectedVars, 'Rows', 'complete');
disp(array2table(correlationMatrix, ...
    'RowNames', {'p', 'Ws1', 'Wd1', 'T1', 'toy'}, ...
    'VariableNames', {'p', 'Ws1', 'Wd1', 'T1', 'toy'}));


% Plot the Correlation Matrix Heatmap
f = figure('Units', 'pixels', 'Position', [600, 300, 800, 500]);
heatmap({'p', 'Ws1', 'Wd1', 'T1', 'toy'}, {'p', 'Ws1', 'Wd1', 'T1', 'toy'}, ...
        correlationMatrix, 'Colormap', jet, 'ColorbarVisible', 'on', 'Interpreter','latex', 'FontSize', 13);
exportgraphics(f,'correlation_heatmap.pdf', 'ContentType', 'vector')



%%

% Define selected variables
selectedVars = {'p', 'Ws1', 'Wd1', 'T1', 'toy'};

% Extract the numeric data for selected variables
dataMatrix = X{:, selectedVars};

% Create figure and set size
f = figure('Units', 'pixels', 'Position', [600, 300, 800, 800]);

% Generate scatter plot matrix
[~, ax, bigAx] = gplotmatrix(dataMatrix, [], [], [39 155 72]/255, '+o', 3, [], 'variable', selectedVars, selectedVars);

% Customize the axes
for i = 1:length(ax)
    for j = 1:length(ax)
        if ~isempty(ax(i, j))
            % Set font size and LaTeX interpreter for tick labels
            set(ax(i, j), 'FontSize', 13, 'TickLabelInterpreter', 'latex');
        end
    end
end

% Increase font size of the diagonal variable names
for i = 1:length(ax)
    if ~isempty(ax(i, i)) % Check if diagonal axes exist
        % Find the diagonal axis
        currentLabel = findobj(ax(i, i), 'Type', 'text');
        if ~isempty(currentLabel)
            % Adjust font size
            set(currentLabel, 'FontSize', 13, 'Interpreter', 'latex'); % Change font size to desired value
        end
    end
end

% Adjust grid and box settings for better visualization
for i = 1:length(ax(:))
    if ~isempty(ax(i))
        grid(ax(i), 'on');
        box(ax(i), 'on');
    end
end
exportgraphics(f,'correlation_plot.pdf', 'ContentType', 'vector')


%% ------------------ Wind direction histogram ----------------------------


% Extract wind direction (angle) and wind speed
windDirection = X.Wd1; % Assuming wind direction is in degrees
windSpeed = X.Ws1;     % Assuming wind speed is for 1-hour ahead

% Create the wind rose
Properties = {'anglenorth',0,'angleeast',90,'labels',{'N (0º)','E (90º)', 'S (180º)', 'W (270º)'}, ...
    'freqlabelangle',45, 'legendfontweight', 'bold', 'axesfontweight','bold', 'frequencyfontweight', 'bold', ...
    'legendfontsize', 13, 'axesfontsize', 13, 'frequencyfontsize', 13, 'legendvariable', 'Ws1', 'titlestring',''};
figHandle = WindRose(windDirection, windSpeed, Properties); 
exportgraphics(figHandle,'wind_rose.pdf', 'ContentType', 'vector')


%% Save script to Github

current_file = 'dataExploration.m';
new_directory = '/home/olivermussmann/Documents/GitHub/02427-AdvancedTimeSeries-CEX4/matlab_code';
copyfile(current_file, new_directory);


