import React, { useState, useEffect } from 'react';
import { getTimeSeriesData, getChangePoints, getPrediction } from '../services/api';
import Chart from './Chart';
import EventHighlight from './EventHighlight';
import Metrics from './Metrics';
import FilterControls from './FilterControls';

const Dashboard = () => {
    const [timeSeriesData, setTimeSeriesData] = useState([]);
    const [changePoints, setChangePoints] = useState([]);
    const [predictionData, setPredictionData] = useState([]);
    
    useEffect(() => {
        fetchTimeSeries();
        fetchChangePoints();
    }, []);
    
    const fetchTimeSeries = async () => {
        const data = await getTimeSeriesData();
        setTimeSeriesData(data);
    };

    const fetchChangePoints = async () => {
        const data = await getChangePoints();
        setChangePoints(data.change_points);
    };

    const handlePrediction = async (startDate, endDate) => {
        const data = await getPrediction(startDate, endDate);
        setPredictionData(data);
    };
    
    return (
        <div>
            <h1>Brent Oil Price Dashboard</h1>
            <FilterControls onPredict={handlePrediction} />
            <Chart data={timeSeriesData} prediction={predictionData} changePoints={changePoints} />
            <EventHighlight events={changePoints} />
            <Metrics />
        </div>
    );
};

export default Dashboard;
