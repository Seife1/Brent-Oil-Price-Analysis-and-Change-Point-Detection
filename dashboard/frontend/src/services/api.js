import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const getTimeSeriesData = async () => {
    const response = await axios.get(`${API_BASE_URL}/time_series`);
    return response.data;
};

export const getChangePoints = async () => {
    const response = await axios.get(`${API_BASE_URL}/change_point_detection`);
    return response.data;
};

export const getCorrelationMatrix = async () => {
    const response = await axios.get(`${API_BASE_URL}/corr_matrix`);
    return response.data;
};

export const getPrediction = async (startDate, endDate) => {
    const response = await axios.post(`${API_BASE_URL}/predict`, { start_date: startDate, end_date: endDate });
    return response.data;
};
