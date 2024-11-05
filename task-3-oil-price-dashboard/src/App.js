import React, { useState, useEffect } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import { TextField, Button, Container, Typography } from '@mui/material';
import './App.css';

function App() {
    const [data, setData] = useState([]);
    const [startDate, setStartDate] = useState('1987-05-20');
    const [endDate, setEndDate] = useState('2022-09-30');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = () => {
        setLoading(true);
        fetch(`http://localhost:5000/api/data?start=${startDate}&end=${endDate}`)
            .then(response => response.json())
            .then(data => {
                setData(data);
                setLoading(false);
            })
            .catch(error => {
                setError(error);
                setLoading(false);
            });
    };

    const handleDateSubmit = (e) => {
        e.preventDefault();
        fetchData();
    };

    return (
        <Container className="App">
            <Typography variant="h4" gutterBottom>
                Brent Oil Prices Dashboard
            </Typography>
            <form onSubmit={handleDateSubmit} className="form">
                <TextField
                    label="Start Date"
                    type="date"
                    value={startDate}
                    onChange={e => setStartDate(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    variant="outlined"
                    style={{ marginRight: '20px' }}
                />
                <TextField
                    label="End Date"
                    type="date"
                    value={endDate}
                    onChange={e => setEndDate(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    variant="outlined"
                />
                <Button type="submit" variant="contained" color="primary" style={{ marginLeft: '20px' }}>
                    Fetch Data
                </Button>
            </form>
            {loading ? <Typography>Loading...</Typography> : (
                <LineChart width={800} height={400} data={data}>
                    <Line type="monotone" dataKey="Price" stroke="#1976d2" />
                    <CartesianGrid stroke="#ccc" />
                    <XAxis dataKey="Date" />
                    <YAxis />
                    <Tooltip />
                </LineChart>
            )}
            {error && <Typography color="error">Error loading data: {error.message}</Typography>}
        </Container>
    );
}

export default App;
 