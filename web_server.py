import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flask import Flask, render_template, jsonify, request
from ai import Dqn, Network, ReplayMemory

app = Flask(__name__)

# Initialize the brain
brain = Dqn(5, 3, 0.9)

# Load the pre-trained model if available
if os.path.isfile('last_brain.pth'):
    print("Loading pre-trained model...")
    brain.load()

# Global variables for simulation
sand = np.zeros((500, 500))
goal_x = 20
goal_y = 480
last_reward = 0
last_distance = 0
car_x = 250
car_y = 250
car_angle = 0
car_velocity = 6
scores = []

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deep Q-Learning Car Simulation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            canvas { border: 1px solid #000; display: block; margin: 20px auto; }
            .controls { text-align: center; margin-bottom: 20px; }
            button { padding: 10px 20px; margin: 0 10px; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Deep Q-Learning Car Simulation</h1>
        <div class="controls">
            <button id="startBtn">Start Simulation</button>
            <button id="clearBtn">Clear Sand</button>
            <button id="saveBtn">Save Brain</button>
        </div>
        <canvas id="simCanvas" width="500" height="500"></canvas>
        <div id="stats" style="text-align: center;">
            <p>Reward: <span id="reward">0</span></p>
            <p>Score: <span id="score">0</span></p>
        </div>
        
        <script>
            const canvas = document.getElementById('simCanvas');
            const ctx = canvas.getContext('2d');
            const startBtn = document.getElementById('startBtn');
            const clearBtn = document.getElementById('clearBtn');
            const saveBtn = document.getElementById('saveBtn');
            const rewardSpan = document.getElementById('reward');
            const scoreSpan = document.getElementById('score');
            
            let isSimulating = false;
            let isDrawing = false;
            let lastX, lastY;
            
            // Draw the initial canvas
            function drawCanvas() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw the sand
                fetch('/get_sand')
                    .then(response => response.json())
                    .then(data => {
                        const sand = data.sand;
                        const imgData = ctx.createImageData(500, 500);
                        
                        for (let y = 0; y < 500; y++) {
                            for (let x = 0; x < 500; x++) {
                                const idx = (y * 500 + x) * 4;
                                if (sand[y][x] > 0) {
                                    imgData.data[idx] = 204;     // R
                                    imgData.data[idx + 1] = 178; // G
                                    imgData.data[idx + 2] = 0;   // B
                                    imgData.data[idx + 3] = 255; // A
                                } else {
                                    imgData.data[idx] = 255;     // R
                                    imgData.data[idx + 1] = 255; // G
                                    imgData.data[idx + 2] = 255; // B
                                    imgData.data[idx + 3] = 255; // A
                                }
                            }
                        }
                        
                        ctx.putImageData(imgData, 0, 0);
                        
                        // Draw the car
                        ctx.save();
                        ctx.translate(data.car_x, data.car_y);
                        ctx.rotate(data.car_angle * Math.PI / 180);
                        ctx.fillStyle = 'blue';
                        ctx.fillRect(-10, -5, 20, 10);
                        ctx.restore();
                        
                        // Draw the goal
                        ctx.fillStyle = 'green';
                        ctx.beginPath();
                        ctx.arc(data.goal_x, data.goal_y, 10, 0, 2 * Math.PI);
                        ctx.fill();
                        
                        // Update stats
                        rewardSpan.textContent = data.last_reward.toFixed(2);
                        scoreSpan.textContent = data.score.toFixed(2);
                    });
            }
            
            // Start the simulation
            startBtn.addEventListener('click', () => {
                isSimulating = !isSimulating;
                startBtn.textContent = isSimulating ? 'Pause Simulation' : 'Start Simulation';
                
                if (isSimulating) {
                    simulationLoop();
                }
            });
            
            // Clear the sand
            clearBtn.addEventListener('click', () => {
                fetch('/clear_sand')
                    .then(() => drawCanvas());
            });
            
            // Save the brain
            saveBtn.addEventListener('click', () => {
                fetch('/save_brain')
                    .then(response => response.json())
                    .then(data => {
                        alert('Brain saved successfully!');
                    });
            });
            
            // Simulation loop
            function simulationLoop() {
                if (!isSimulating) return;
                
                fetch('/step')
                    .then(response => response.json())
                    .then(data => {
                        drawCanvas();
                        setTimeout(simulationLoop, 50);
                    });
            }
            
            // Drawing sand on the canvas
            canvas.addEventListener('mousedown', (e) => {
                isDrawing = true;
                lastX = e.offsetX;
                lastY = e.offsetY;
                
                // Add sand at the clicked position
                fetch('/add_sand', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        x: lastX,
                        y: lastY,
                    }),
                }).then(() => drawCanvas());
            });
            
            canvas.addEventListener('mousemove', (e) => {
                if (!isDrawing) return;
                
                const x = e.offsetX;
                const y = e.offsetY;
                
                // Add sand along the line
                fetch('/add_sand_line', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        x1: lastX,
                        y1: lastY,
                        x2: x,
                        y2: y,
                    }),
                }).then(() => {
                    lastX = x;
                    lastY = y;
                    drawCanvas();
                });
            });
            
            canvas.addEventListener('mouseup', () => {
                isDrawing = false;
            });
            
            // Initial draw
            drawCanvas();
        </script>
    </body>
    </html>
    """

@app.route('/get_sand')
def get_sand():
    global sand, car_x, car_y, car_angle, goal_x, goal_y, last_reward, scores
    
    return jsonify({
        'sand': sand.tolist(),
        'car_x': car_x,
        'car_y': car_y,
        'car_angle': car_angle,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'last_reward': float(last_reward),
        'score': float(brain.score())
    })

@app.route('/step')
def step():
    global sand, car_x, car_y, car_angle, car_velocity, goal_x, goal_y, last_reward, last_distance, scores
    
    # Calculate orientation
    xx = goal_x - car_x
    yy = goal_y - car_y
    orientation = np.arctan2(yy, xx) * 180 / np.pi - car_angle
    orientation = orientation / 180.0
    
    # Get signals (simplified)
    signal1 = 0
    signal2 = 0
    signal3 = 0
    
    # Check if sensors detect sand
    sensor1_x = car_x + 30 * np.cos(car_angle * np.pi / 180)
    sensor1_y = car_y + 30 * np.sin(car_angle * np.pi / 180)
    
    sensor2_x = car_x + 30 * np.cos((car_angle + 30) * np.pi / 180)
    sensor2_y = car_y + 30 * np.sin((car_angle + 30) * np.pi / 180)
    
    sensor3_x = car_x + 30 * np.cos((car_angle - 30) * np.pi / 180)
    sensor3_y = car_y + 30 * np.sin((car_angle - 30) * np.pi / 180)
    
    # Check boundaries
    if 10 <= sensor1_x < 490 and 10 <= sensor1_y < 490:
        signal1 = np.sum(sand[int(sensor1_y)-10:int(sensor1_y)+10, int(sensor1_x)-10:int(sensor1_x)+10]) / 400.0
    else:
        signal1 = 1.0
        
    if 10 <= sensor2_x < 490 and 10 <= sensor2_y < 490:
        signal2 = np.sum(sand[int(sensor2_y)-10:int(sensor2_y)+10, int(sensor2_x)-10:int(sensor2_x)+10]) / 400.0
    else:
        signal2 = 1.0
        
    if 10 <= sensor3_x < 490 and 10 <= sensor3_y < 490:
        signal3 = np.sum(sand[int(sensor3_y)-10:int(sensor3_y)+10, int(sensor3_x)-10:int(sensor3_x)+10]) / 400.0
    else:
        signal3 = 1.0
    
    # Create the signal array
    last_signal = [signal1, signal2, signal3, orientation, -orientation]
    
    # Get action from the brain
    action = brain.update(last_reward, last_signal)
    scores.append(brain.score())
    
    # Convert action to rotation
    action2rotation = [0, 20, -20]
    rotation = action2rotation[action]
    
    # Update car angle
    car_angle += rotation
    
    # Calculate distance to goal
    distance = np.sqrt((car_x - goal_x)**2 + (car_y - goal_y)**2)
    
    # Update car position
    if 0 <= car_x < 500 and 0 <= car_y < 500 and 0 <= int(car_x) < 500 and 0 <= int(car_y) < 500:
        if sand[int(car_y), int(car_x)] > 0:
            car_velocity = 1
            last_reward = -1
        else:
            car_velocity = 6
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1
    
    # Move the car
    car_x += car_velocity * np.cos(car_angle * np.pi / 180)
    car_y += car_velocity * np.sin(car_angle * np.pi / 180)
    
    # Check boundaries
    if car_x < 10:
        car_x = 10
        last_reward = -1
    if car_x > 490:
        car_x = 490
        last_reward = -1
    if car_y < 10:
        car_y = 10
        last_reward = -1
    if car_y > 490:
        car_y = 490
        last_reward = -1
    
    # Check if goal reached
    if distance < 30:
        if goal_x == 20 and goal_y == 480:
            goal_x = 480
            goal_y = 20
        else:
            goal_x = 20
            goal_y = 480
    
    last_distance = distance
    
    return jsonify({'status': 'success'})

@app.route('/clear_sand')
def clear_sand():
    global sand
    sand = np.zeros((500, 500))
    return jsonify({'status': 'success'})

@app.route('/add_sand', methods=['POST'])
def add_sand():
    global sand
    data = request.json
    x = int(data['x'])
    y = int(data['y'])
    
    if 0 <= x < 500 and 0 <= y < 500:
        sand[max(0, y-10):min(500, y+10), max(0, x-10):min(500, x+10)] = 1
    
    return jsonify({'status': 'success'})

@app.route('/add_sand_line', methods=['POST'])
def add_sand_line():
    global sand
    data = request.json
    x1 = int(data['x1'])
    y1 = int(data['y1'])
    x2 = int(data['x2'])
    y2 = int(data['y2'])
    
    # Draw a line using Bresenham's algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        if 0 <= x1 < 500 and 0 <= y1 < 500:
            sand[max(0, y1-10):min(500, y1+10), max(0, x1-10):min(500, x1+10)] = 1
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return jsonify({'status': 'success'})

@app.route('/save_brain')
def save_brain():
    brain.save()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000, debug=True)