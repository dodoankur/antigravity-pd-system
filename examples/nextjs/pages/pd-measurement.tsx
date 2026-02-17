/**
 * PD Measurement Page - Next.js Integration Example
 *
 * This example shows how to integrate the PD Measurement Widget
 * into a Next.js application.
 *
 * Installation:
 * 1. Copy the PDMeasurer component to your components directory
 * 2. Copy the styles.css to your styles directory
 * 3. Import and use as shown below
 */

import { useState } from "react";
import Head from "next/head";
// In a real Next.js app, you would import from your components directory:
// import { PDMeasurer } from '@/components/PDMeasurer';
// import '@/components/PDMeasurer/styles.css';

// For this example, we'll show the expected shape of the component usage

interface PDMeasurementResult {
    overall_pd_mm: number;
    left_pd_mm: number;
    right_pd_mm: number;
    method: "iris_estimation" | "reference_object";
    model_used: string;
    confidence_score: number;
    error_margin: {
        value_mm: number;
        percentage: number;
        confidence_score: number;
    };
    disclaimer: string;
}

export default function PDMeasurementPage() {
    const [lastResult, setLastResult] = useState<PDMeasurementResult | null>(null);
    const [showResult, setShowResult] = useState(false);

    const handleMeasurement = (result: PDMeasurementResult) => {
        console.log("PD Measurement Result:", result);
        setLastResult(result);
        setShowResult(true);

        // You can save to your database, send to analytics, etc.
        // Example: saveToDatabase(result);
    };

    const handleError = (error: string) => {
        console.error("PD Measurement Error:", error);
        // Handle error (show toast, log to error tracking, etc.)
    };

    return (
        <>
            <Head>
                <title>Measure Your PD | Eyewear Shop</title>
                <meta name="description" content="Measure your pupil distance for accurate eyewear fitting" />
            </Head>

            <main className="min-h-screen bg-gradient-to-br from-indigo-600 to-purple-700 py-12 px-4">
                <div className="max-w-xl mx-auto">
                    <header className="text-center text-white mb-8">
                        <h1 className="text-3xl font-bold mb-2">Measure Your Pupil Distance</h1>
                        <p className="opacity-90">Get accurate PD measurements for your prescription eyewear</p>
                    </header>

                    {/* PD Measurement Widget */}
                    <div className="mb-6">
                        {/* 
              In your actual Next.js app, use:
              
              <PDMeasurer
                apiEndpoint={process.env.NEXT_PUBLIC_PD_API_ENDPOINT || '/api/pd/measure'}
                onMeasurement={handleMeasurement}
                onError={handleError}
              />
            */}
                        <div className="bg-white rounded-xl shadow-xl p-6 text-center">
                            <p className="text-gray-600 mb-4">To use this component:</p>
                            <ol className="text-left text-sm text-gray-700 space-y-2">
                                <li>1. Install the PD widget package or copy the component files</li>
                                <li>2. Import PDMeasurer and its styles</li>
                                <li>3. Configure your API endpoint</li>
                                <li>4. Handle the onMeasurement callback</li>
                            </ol>
                            <pre className="mt-4 bg-gray-100 p-4 rounded-lg text-xs text-left overflow-x-auto">
                                {`import { PDMeasurer } from '@/components/PDMeasurer';
import '@/components/PDMeasurer/styles.css';

<PDMeasurer
  apiEndpoint="/api/pd/measure"
  onMeasurement={(result) => {
    console.log('PD:', result.overall_pd_mm);
  }}
  onError={(error) => {
    console.error(error);
  }}
/>`}
                            </pre>
                        </div>
                    </div>

                    {/* Result Display */}
                    {showResult && lastResult && (
                        <div className="bg-white rounded-xl shadow-xl p-6">
                            <h3 className="font-semibold text-gray-900 mb-4">Measurement Result</h3>
                            <div className="grid grid-cols-3 gap-4 mb-4">
                                <div className="text-center">
                                    <div className="text-xs text-gray-500 uppercase">Overall PD</div>
                                    <div className="text-2xl font-bold text-indigo-600">{lastResult.overall_pd_mm}mm</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-xs text-gray-500 uppercase">Left</div>
                                    <div className="text-2xl font-bold text-indigo-600">{lastResult.left_pd_mm}mm</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-xs text-gray-500 uppercase">Right</div>
                                    <div className="text-2xl font-bold text-indigo-600">{lastResult.right_pd_mm}mm</div>
                                </div>
                            </div>
                            <div className="text-sm text-gray-600">
                                <strong>Confidence:</strong> {(lastResult.confidence_score * 100).toFixed(0)}%
                                <br />
                                <strong>Accuracy:</strong> ±{lastResult.error_margin.value_mm}mm
                            </div>
                        </div>
                    )}

                    <footer className="text-center text-white/70 text-sm mt-8">
                        <p>For prescription eyewear, verify with your optician</p>
                    </footer>
                </div>
            </main>
        </>
    );
}
