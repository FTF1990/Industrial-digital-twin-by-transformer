#!/usr/bin/env python3
"""
WebSocket Streaming Inference Client Demo

This script demonstrates how to use the WebSocket streaming inference API.
"""
import asyncio
import websockets
import json
import time
from typing import Dict, Any


class StreamInferenceClient:
    """Python client for WebSocket streaming inference"""
    
    def __init__(self, uri: str = "ws://localhost:8000/api/v1/inference/stream"):
        self.uri = uri
        self.websocket = None
        self.session_id = None
        
    async def connect(self, ensemble_name: str, mode: str = "single", **config_options):
        """
        Connect to WebSocket server and configure session
        
        Args:
            ensemble_name: Name of ensemble model to use
            mode: "single" or "batch"
            **config_options: Additional configuration options
        """
        self.websocket = await websockets.connect(self.uri)
        print(f"✅ Connected to {self.uri}")
        
        # Send configuration
        config_msg = {
            "type": "config",
            "data": {
                "ensemble_name": ensemble_name,
                "mode": mode,
                **config_options
            }
        }
        
        await self.websocket.send(json.dumps(config_msg))
        print(f"📤 Sent configuration: mode={mode}, ensemble={ensemble_name}")
        
        # Wait for acknowledgment
        response = await self.websocket.recv()
        response_data = json.loads(response)
        
        if response_data.get('type') == 'config_ack' and response_data.get('status') == 'success':
            self.session_id = response_data['session_id']
            print(f"✅ Configuration acknowledged")
            print(f"   Session ID: {self.session_id}")
            print(f"   Ensemble: {response_data['ensemble_info']['ensemble_name']}")
            print(f"   Signals: {response_data['ensemble_info']['num_signals']}")
            print(f"   Using Residual Boost: {response_data['ensemble_info']['signals_using_boost']}")
            return True
        else:
            print(f"❌ Configuration failed: {response_data}")
            return False
    
    async def predict_single(self, boundary_signals: Dict[str, float], timestamp: str = None):
        """
        Send single prediction request
        
        Args:
            boundary_signals: Dict of boundary signal values
            timestamp: Optional timestamp
            
        Returns:
            Prediction response
        """
        predict_msg = {
            "type": "predict",
            "data": {
                "boundary_signals": boundary_signals,
                "timestamp": timestamp
            }
        }
        
        await self.websocket.send(json.dumps(predict_msg))
        
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def predict_batch(self, batch: list, timestamps: list = None):
        """
        Send batch prediction request
        
        Args:
            batch: List of boundary signal dicts
            timestamps: Optional list of timestamps
            
        Returns:
            Batch prediction response
        """
        predict_msg = {
            "type": "predict_batch",
            "data": {
                "batch": batch,
                "timestamps": timestamps
            }
        }
        
        await self.websocket.send(json.dumps(predict_msg))
        
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def ping(self):
        """Send ping to check connection"""
        await self.websocket.send(json.dumps({"type": "ping"}))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            print(f"🔌 Connection closed")


async def demo_single_mode():
    """Demo: Single sample mode"""
    print("\n" + "=" * 80)
    print("Demo 1: Single Sample Mode")
    print("=" * 80)
    
    client = StreamInferenceClient()
    
    # UPDATE THIS WITH YOUR ENSEMBLE NAME
    ensemble_name = "Ensemble_your_model_20251215_103000"
    
    try:
        # Connect
        connected = await client.connect(ensemble_name, mode="single")
        
        if not connected:
            print("Failed to connect. Please check:")
            print("1. Server is running (python -m fastapi_inference.main)")
            print("2. Ensemble model exists")
            return
        
        # Send 10 predictions
        print("\n📊 Sending 10 predictions...")
        
        for i in range(10):
            # Example boundary signals - UPDATE WITH YOUR ACTUAL SIGNALS
            boundary_signals = {
                "Temperature_boundary_1": 23.5 + i * 0.1,
                "Pressure_boundary_1": 101.3 + i * 0.05,
                "Flow_boundary_1": 50.0 + i * 0.2,
                # Add all your boundary signals here
            }
            
            # Send prediction request
            response = await client.predict_single(boundary_signals)
            
            if response.get('type') == 'prediction' and response.get('status') == 'success':
                data = response['data']
                print(f"\n✅ Prediction {i+1}:")
                print(f"   Latency: {data['latency_ms']:.2f} ms")
                print(f"   Predictions: {list(data['predictions'].keys())[:3]}... ({len(data['predictions'])} signals)")
                print(f"   Boost used: {len(data['signals_used_boost'])} signals")
            else:
                print(f"❌ Prediction {i+1} failed: {response}")
            
            await asyncio.sleep(0.1)  # 100ms interval
        
        # Test ping
        print("\n🏓 Testing ping...")
        pong = await client.ping()
        print(f"✅ Pong received: {pong['timestamp']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()


async def demo_batch_mode():
    """Demo: Batch mode"""
    print("\n" + "=" * 80)
    print("Demo 2: Batch Mode")
    print("=" * 80)
    
    client = StreamInferenceClient()
    
    # UPDATE THIS WITH YOUR ENSEMBLE NAME
    ensemble_name = "Ensemble_your_model_20251215_103000"
    
    try:
        # Connect with batch configuration
        connected = await client.connect(
            ensemble_name,
            mode="batch",
            batch_size=20
        )
        
        if not connected:
            print("Failed to connect")
            return
        
        # Send batch predictions
        print("\n📊 Sending batch predictions...")
        
        # Create batch of 10 samples
        batch = []
        for i in range(10):
            # Example boundary signals - UPDATE WITH YOUR ACTUAL SIGNALS
            sample = {
                "Temperature_boundary_1": 23.5 + i * 0.1,
                "Pressure_boundary_1": 101.3 + i * 0.05,
                "Flow_boundary_1": 50.0 + i * 0.2,
                # Add all your boundary signals here
            }
            batch.append(sample)
        
        # Send batch
        response = await client.predict_batch(batch)
        
        if response.get('type') == 'prediction_batch' and response.get('status') == 'success':
            data = response['data']
            print(f"\n✅ Batch prediction successful:")
            print(f"   Samples: {data['count']}")
            print(f"   Total latency: {data['latency_ms']:.2f} ms")
            print(f"   Average latency: {data['latency_ms'] / data['count']:.2f} ms per sample")
            print(f"   Predictions shape: {len(data['predictions'])} samples x {len(data['predictions'][0])} signals")
        else:
            print(f"❌ Batch prediction failed: {response}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()


async def demo_continuous_stream():
    """Demo: Continuous data stream"""
    print("\n" + "=" * 80)
    print("Demo 3: Continuous Data Stream")
    print("=" * 80)
    
    client = StreamInferenceClient()
    
    # UPDATE THIS WITH YOUR ENSEMBLE NAME
    ensemble_name = "Ensemble_your_model_20251215_103000"
    
    try:
        connected = await client.connect(ensemble_name, mode="single")
        
        if not connected:
            print("Failed to connect")
            return
        
        print("\n📊 Simulating continuous data stream (30 seconds)...")
        print("   Press Ctrl+C to stop")
        
        start_time = time.time()
        prediction_count = 0
        total_latency = 0.0
        
        # Stream for 30 seconds
        while time.time() - start_time < 30:
            # Simulate sensor readings
            boundary_signals = {
                "Temperature_boundary_1": 23.0 + (time.time() % 10) * 0.5,
                "Pressure_boundary_1": 101.0 + (time.time() % 5) * 0.2,
                "Flow_boundary_1": 50.0 + (time.time() % 8) * 0.3,
                # Add all your boundary signals here
            }
            
            response = await client.predict_single(boundary_signals)
            
            if response.get('type') == 'prediction':
                prediction_count += 1
                total_latency += response['data']['latency_ms']
                
                if prediction_count % 10 == 0:
                    avg_latency = total_latency / prediction_count
                    print(f"   Predictions: {prediction_count}, Avg latency: {avg_latency:.2f} ms")
            
            await asyncio.sleep(0.05)  # 50ms = 20 Hz
        
        print(f"\n✅ Streaming completed:")
        print(f"   Total predictions: {prediction_count}")
        print(f"   Average latency: {total_latency / prediction_count:.2f} ms")
        print(f"   Throughput: {prediction_count / 30:.1f} predictions/sec")
    
    except KeyboardInterrupt:
        print("\n⚠️  Streaming interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()


def main():
    """Main function"""
    print("=" * 80)
    print("WebSocket Streaming Inference Client Demo")
    print("=" * 80)
    
    print("\n⚠️  IMPORTANT: Before running, update the following:")
    print("1. ensemble_name in each demo function")
    print("2. boundary_signals dict with your actual signal names")
    print("3. Make sure the server is running: python -m fastapi_inference.main")
    
    print("\nAvailable demos:")
    print("1. Single sample mode (10 predictions)")
    print("2. Batch mode (1 batch of 10 samples)")
    print("3. Continuous stream (30 seconds)")
    
    choice = input("\nSelect demo (1/2/3 or 'all'): ").strip()
    
    if choice == '1':
        asyncio.run(demo_single_mode())
    elif choice == '2':
        asyncio.run(demo_batch_mode())
    elif choice == '3':
        asyncio.run(demo_continuous_stream())
    elif choice.lower() == 'all':
        asyncio.run(demo_single_mode())
        asyncio.run(demo_batch_mode())
        asyncio.run(demo_continuous_stream())
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
