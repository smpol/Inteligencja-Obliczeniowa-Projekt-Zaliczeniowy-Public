// Config import removed to fix build error
// Actually the previous api.js didn't use config, it used relative /predict for proxy.
// Let's stick to the pattern that works with the proxy setup.
// Let's stick to the pattern that works with the proxy setup.

const API_BASE = ''; // Proxy handles the host

export const sendPrediction = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });
    
    // Check content type for JSON
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        console.error("Non-JSON response:", text);
        throw new Error(`Server returned ${response.status} ${response.statusText} (non-JSON)`);
    }

    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || `Server error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
};

export const compareModels = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE}/predict_all`, {
      method: 'POST',
      body: formData,
    });
    
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
         return null; 
    }

    if (!response.ok) {
        return null;
    }

    return await response.json();
  } catch (error) {
    console.error("Comparison API Error:", error);
    return null;
  }
};
