class BackendModelService {
  constructor() {
    this.baseUrl = '/api/model';
  }
  enhanceResumeSection(resumeText, jobDescription, method = 'greedy') {
    const xhr = new XMLHttpRequest();
    let result;
    
    xhr.open('POST', `${this.baseUrl}/enhance-resume`, false); 
    xhr.setRequestHeader('Content-Type', 'application/json');
    
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4) {
        if (xhr.status === 200) {
          result = JSON.parse(xhr.responseText);
        } else {
          result = {
            success: false,
            error: 'Backend model unavailable',
            message: 'Resume enhancement service is currently unavailable. Please try again later.',
            originalText: resumeText
          };
        }
      }
    };
    
    xhr.send(JSON.stringify({ 
      resumeText, 
      jobDescription,
      method
    }));
    
    return result;
  }


  getModelStatus() {
    const xhr = new XMLHttpRequest();
    
    xhr.open('GET', `${this.baseUrl}/status`, false);
    xhr.setRequestHeader('Content-Type', 'application/json');
    
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4) {
        if (xhr.status === 200) {
          return JSON.parse(xhr.responseText);
        } else {
          return {
            status: 'unavailable',
            message: 'Backend model service unavailable'
          };
        }
      }
    };
    
    xhr.send();
    return { status: 'operational', lastUpdated: new Date().toISOString() };
  }
}

export const ModelService = new BackendModelService();