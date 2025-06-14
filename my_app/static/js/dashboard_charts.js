function showModal(content) {
    document.getElementById('content-text').innerText = content;
    $('#contentModal').modal('show');
}

function selectAll(source) {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    for (let i = 0; i < checkboxes.length; i++) {
        if (checkboxes[i] !== source)
            checkboxes[i].checked = source.checked;
    }
}

function initializeCharts(scoresTR, scoresCC, scoresLR, scoresGRA, creationTimes) {
    const essayNumbers = [...Array(scoresTR.length).keys()].map(x => x + 1); // Array from 1 to the length of scores

    const rubricChartData = {
        labels: essayNumbers,
        datasets: [
            {
                label: 'Task Response',
                data: scoresTR,
                borderColor: 'rgba(54, 162, 235, 1.0)',
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                fill: false,
                pointRadius: 5
            },
            {
                label: 'Coherence and Cohesion',
                data: scoresCC,
                borderColor: 'rgba(255, 99, 132, 1.0)',
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                fill: false,
                pointRadius: 5
            },
            {
                label: 'Lexical Resource',
                data: scoresLR,
                borderColor: 'rgba(75, 192, 192, 1.0)',
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                fill: false,
                pointRadius: 5
            },
            {
                label: 'Grammatical Range and Accuracy',
                data: scoresGRA,
                borderColor: 'rgba(153, 102, 255, 1.0)',
                backgroundColor: 'rgba(153, 102, 255, 0.6)',
                fill: false,
                pointRadius: 5
            }
        ]
    };

    const rubricChartOptions = {
        responsive: true,
        plugins: {
            tooltip: {
                callbacks: {
                    label: function (tooltipItem) {
                        const index = tooltipItem.dataIndex;
                        switch (tooltipItem.dataset.label) {
                            case 'Task Response':
                                return `Task Response Score: ${tooltipItem.raw}, Time: ${creationTimes[index]}`;
                            case 'Coherence and Cohesion':
                                return `Coherence and Cohesion Score: ${tooltipItem.raw}, Time: ${creationTimes[index]}`;
                            case 'Lexical Resource':
                                return `Lexical Resource Score: ${tooltipItem.raw}, Time: ${creationTimes[index]}`;
                            case 'Grammatical Range and Accuracy':
                                return `Grammatical Range and Accuracy Score: ${tooltipItem.raw}, Time: ${creationTimes[index]}`;
                            default:
                                return `Score: ${tooltipItem.raw}, Time: ${creationTimes[index]}`;
                        }
                    }
                }
            },
            legend: {
                display: true,
                labels: {
                    color: 'white',
                    padding: 20
                },
                position: 'top',
                align: 'center'
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Essay Number',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    display: true
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Score',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                min: 0,
                max: 10,
                grid: {
                    display: true,
                    color: 'rgba(255, 255, 255, 0.2)'
                }
            }
        }
    };

    new Chart(document.getElementById('RubricChart'), {
        type: 'line',
        data: rubricChartData,
        options: rubricChartOptions
    });

    const essaysPerDay = {};
    creationTimes.forEach(time => {
        const date = time.split(' ')[0];
        essaysPerDay[date] = (essaysPerDay[date] || 0) + 1;
    });
    const sortedEssaysArray = Object.entries(essaysPerDay).sort((a, b) => new Date(a[0]) - new Date(b[0]));
    const sortedEssaysPerDay = Object.fromEntries(sortedEssaysArray);

    const dates = Object.keys(sortedEssaysPerDay);
    const essayCounts = Object.values(sortedEssaysPerDay);

    const essayCountChartData = {
        labels: dates,
        datasets: [{
            label: 'Essays Per Day',
            data: essayCounts,
            borderColor: 'rgba(255, 165, 0, 1.0)',
            backgroundColor: 'rgba(145, 128, 128, 0.5)', 
            fill: true,
            pointRadius: 5
        }]
    };

    const essayCountChartOptions = {
        responsive: true,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Date',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    display: true
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Number of Essays',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    display: true,
                    color: 'rgba(255, 255, 255, 0.2)' // Light grid lines
                }
            }
        }
    };
    new Chart(document.getElementById('EssayCountChart'), {
        type: 'bar',
        data: essayCountChartData,
        options: essayCountChartOptions
    });
}
