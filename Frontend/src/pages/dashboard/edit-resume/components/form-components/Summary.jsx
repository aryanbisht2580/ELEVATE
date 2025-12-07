
import React, { useState } from "react";
import { Sparkles, LoaderCircle, FileText, Zap, User, Award, TrendingUp, Briefcase, Wrench, FolderOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useDispatch } from "react-redux";
import { addResumeData } from "@/features/resume/resumeFeatures";
import { useParams } from "react-router-dom";
import { toast } from "sonner";
import { AIChatSession } from "@/Services/AiModel";
import { updateThisResume } from "@/Services/resumeAPI";

const prompt =
  "Job Title: {jobTitle} , Depends on job title give me list of summery for 3 experience level, Mid Level and Freasher level in 3 -4 lines in array format, With summery and experience_level Field in JSON Format";

function Summary({ resumeInfo, enanbledNext, enanbledPrev }) {
  const dispatch = useDispatch();
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState(resumeInfo?.summary || "");
  const [aiGeneratedSummeryList, setAiGenerateSummeryList] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const { resume_id } = useParams();

  const handleInputChange = (e) => {
    enanbledNext(false);
    enanbledPrev(false);
    dispatch(
      addResumeData({
        ...resumeInfo,
        [e.target.name]: e.target.value,
      })
    );
    setSummary(e.target.value);
  };

  const onSave = (e) => {
    e.preventDefault();
    setLoading(true);
    console.log("Started Saving Summary");
    const data = {
      data: { summary },
    };
    if (resume_id) {
      updateThisResume(resume_id, data)
        .then((data) => {
          toast("Resume Updated", "success");
        })
        .catch((error) => {
          toast("Error updating resume", `${error.message}`);
        })
        .finally(() => {
          enanbledNext(true);
          enanbledPrev(true);
          setLoading(false);
        });
    }
  };

  const setSummery = (summary) => {
    dispatch(
      addResumeData({
        ...resumeInfo,
        summary: summary,
      })
    );
    setSummary(summary);
  };

  const GenerateSummeryFromAI = async () => {
    setLoading(true);
    console.log("Generate Summery From AI for", resumeInfo?.jobTitle);
    if (!resumeInfo?.jobTitle) {
      toast("Please Add Job Title");
      setLoading(false);
      return;
    }
    const PROMPT = prompt.replace("{jobTitle}", resumeInfo?.jobTitle);
    try {
      const result = await AIChatSession.sendMessage(PROMPT);
      console.log(JSON.parse(result.response.text()));
      setAiGenerateSummeryList(JSON.parse(result.response.text()));
      toast("Summery Generated", "success");
    } catch (error) {
      console.log(error);
      toast(`${error.message}`, `${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getExperienceIcon = (level) => {
    const levelLower = level?.toLowerCase();
    if (levelLower?.includes('fresh')) return User;
    if (levelLower?.includes('mid')) return TrendingUp;
    return Award;
  };

  // Function to generate content based on job description
  const generateContentFromJobDescription = async () => {
    if (!jobDescription.trim()) {
      toast("Please enter a job description first");
      return;
    }
    
    setLoading(true);
    try {
      // Prompt to generate professional summary based on job description
      const summaryPrompt = `Based on the following job description: "${jobDescription}", generate a professional summary that highlights relevant skills and experience for this role. Keep it concise in 2-3 sentences. Use information from the user's resume: ${JSON.stringify(resumeInfo)}. Only use skills and experiences that match the given resume data. Prioritize skills and experiences that are mentioned in the job description.`;
      const summaryResult = await AIChatSession.sendMessage(summaryPrompt);
      const generatedSummary = summaryResult.response.text();
      
      // Dispatch the new summary to update the form
      setSummery(generatedSummary);
      
      // Update Experience sections to match the job description
      if (resumeInfo?.experience && resumeInfo.experience.length > 0) {
        const experiencePromises = resumeInfo.experience.map(async (exp, index) => {
          const experiencePrompt = `Based on this job description: "${jobDescription}", optimize the following work experience to better match the role: Position: "${exp.title}", Company: "${exp.companyName}", Description: "${exp.workSummary}". Rewrite the workSummary to emphasize relevant skills and achievements that align with the job description, using action verbs and quantifiable results where possible. Keep the existing position title and company name, but modify the workSummary. Only use skills and experiences that match the given resume data. Format as JSON with title, companyName, and workSummary fields: {"title": "...", "companyName": "...", "workSummary": "...", "city": "...", "state": "...", "startDate": "...", "endDate": "...", "currentlyWorking": "..."} Use the original values for fields not mentioned in the prompt.`;
          const expResult = await AIChatSession.sendMessage(experiencePrompt);
          let updatedExp;
          
          try {
            updatedExp = JSON.parse(expResult.response.text());
          } catch (e) {
            // If parsing fails, try to extract the work summary content
            updatedExp = {
              ...exp,
              workSummary: expResult.response.text()
            };
          }
          
          return updatedExp;
        });
        
        const updatedExperiences = await Promise.all(experiencePromises);
        dispatch(
          addResumeData({
            ...resumeInfo,
            experience: updatedExperiences
          })
        );
      }
      
      // Update Project sections to match the job description
      if (resumeInfo?.projects && resumeInfo.projects.length > 0) {
        const projectPromises = resumeInfo.projects.map(async (proj, index) => {
          const projectPrompt = `Based on this job description: "${jobDescription}", optimize the following project to better match the role: Project: "${proj.projectName}", Tech Stack: "${proj.techStack}", Summary: "${proj.projectSummary}". Rewrite the project summary to emphasize technologies and outcomes that align with the job description. Keep the project name and technology stack, but modify the projectSummary to better match the requirements. Only use skills and experiences that match the given resume data. Format as JSON with projectName, techStack, and projectSummary fields: {"projectName": "...", "techStack": "...", "projectSummary": "..."} Use the original values for fields not mentioned in the prompt.`;
          const projResult = await AIChatSession.sendMessage(projectPrompt);
          let updatedProj;
          
          try {
            updatedProj = JSON.parse(projResult.response.text());
          } catch (e) {
            // If parsing fails, try to extract the project summary content
            updatedProj = {
              ...proj,
              projectSummary: projResult.response.text()
            };
          }
          
          return updatedProj;
        });
        
        const updatedProjects = await Promise.all(projectPromises);
        dispatch(
          addResumeData({
            ...resumeInfo,
            projects: updatedProjects
          })
        );
      }
      
      toast("Professional Summary, Experience and Projects updated based on job description", "success");
    } catch (error) {
      console.log(error);
      toast(`${error.message}`, `${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      
      <div className="relative bg-white border-2 border-gray-200 rounded-2xl shadow-lg overflow-hidden">
        
        <div className="relative bg-gradient-to-r from-gray-900 to-black p-8 pb-12">
          
          <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-5 rounded-full -translate-y-16 translate-x-16"></div>
          <div className="absolute bottom-0 left-0 w-24 h-24 bg-white opacity-5 rounded-full translate-y-12 -translate-x-12"></div>
          
          <div className="relative flex items-center space-x-4">
            <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-lg">
              <FileText className="w-8 h-8 text-black" strokeWidth={2} />
            </div>
            <div className="text-white">
              <h2 className="text-2xl font-bold">Professional Summary</h2>
              <p className="text-gray-300 mt-1">Craft your professional narrative</p>
            </div>
          </div>
        </div>

        
        <div className="p-8 -mt-6 relative">
          <div className="bg-white rounded-xl border border-gray-100 p-6 shadow-sm">
            <form className="space-y-6" onSubmit={onSave}>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                    <FileText className="w-4 h-4 text-gray-500" strokeWidth={2} />
                    <span>Professional Summary</span>
                    <span className="text-red-500">*</span>
                  </label>
                  <Button
                    variant="outline"
                    onClick={() => GenerateSummeryFromAI()}
                    type="button"
                    size="sm"
                    disabled={loading}
                    className="border-2 border-black text-black hover:bg-black hover:text-white transition-all duration-200 rounded-lg px-4 py-2 text-sm font-medium"
                  >
                    {loading ? (
                      <LoaderCircle className="h-4 w-4 animate-spin mr-2" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-2" />
                    )}
                    {loading ? "Generating..." : "Generate with AI"}
                  </Button>
                </div>
                
                <div className="relative group">
                  <Textarea
                    name="summary"
                    required
                    value={summary ? summary : resumeInfo?.summary}
                    onChange={handleInputChange}
                    placeholder="Write a compelling summary that highlights your key strengths, experience, and career objectives..."
                    className="min-h-[120px] w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black focus:ring-0 transition-all duration-200 resize-none"
                  />
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gray-200 rounded-full">
                    <div className="h-full bg-black rounded-full transition-all duration-300 scale-x-0 group-focus-within:scale-x-100 origin-left"></div>
                  </div>
                </div>
                
                {/* Job Description Input Field */}
                <div className="space-y-4 mt-6 border-t border-gray-100 pt-6">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <Briefcase className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>Job Description</span>
                    </label>
                    <Button
                      variant="outline"
                      onClick={generateContentFromJobDescription}
                      type="button"
                      size="sm"
                      disabled={loading}
                      className="border-2 border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white transition-all duration-200 rounded-lg px-4 py-2 text-sm font-medium"
                    >
                      {loading ? (
                        <LoaderCircle className="h-4 w-4 animate-spin mr-2" />
                      ) : (
                        <Sparkles className="h-4 w-4 mr-2" />
                      )}
                      {loading ? "Updating..." : "Update Content from Job Description"}
                    </Button>
                  </div>
                  
                  <div className="relative group">
                    <Textarea
                      value={jobDescription}
                      onChange={(e) => setJobDescription(e.target.value)}
                      placeholder="Paste the job description here to optimize your Professional Summary, Experience, and Projects to match the role..."
                      className="min-h-[100px] w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-blue-500 focus:ring-0 transition-all duration-200 resize-none"
                    />
                    <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gray-200 rounded-full">
                      <div className="h-full bg-blue-500 rounded-full transition-all duration-300 scale-x-0 group-focus-within:scale-x-100 origin-left"></div>
                    </div>
                  </div>
                </div>
              </div>

              
              <div className="pt-6 border-t border-gray-100">
                <div className="flex justify-between items-center">
                  <div className="text-sm text-gray-500">
                    <span className="text-red-500">*</span> Required field
                  </div>
                  <Button 
                    type="submit" 
                    disabled={loading}
                    className="px-8 py-3 bg-gradient-to-r from-gray-900 to-black text-white hover:from-black hover:to-gray-900 disabled:from-gray-400 disabled:to-gray-500 rounded-lg font-medium transition-all duration-200 shadow-lg hover:shadow-xl min-w-[120px]"
                  >
                    {loading ? (
                      <div className="flex items-center space-x-2">
                        <LoaderCircle className="w-4 h-4 animate-spin" />
                        <span>Saving...</span>
                      </div>
                    ) : (
                      "Save Summary"
                    )}
                  </Button>
                </div>
              </div>
            </form>
          </div>
        </div>

        
        <div className="h-1 bg-gradient-to-r from-gray-900 to-black"></div>
      </div>

      
      {aiGeneratedSummeryList && (
        <div className="bg-white border-2 border-gray-200 rounded-2xl shadow-lg overflow-hidden">
          
          <div className="bg-gradient-to-r from-gray-100 to-gray-50 p-6 border-b border-gray-200">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-black rounded-full flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" strokeWidth={2} />
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900">AI Suggestions</h2>
                <p className="text-sm text-gray-600">Choose a summary that matches your experience level</p>
              </div>
            </div>
          </div>
          
          <div className="p-6 space-y-4">
            {aiGeneratedSummeryList?.map((item, index) => {
              const IconComponent = getExperienceIcon(item?.experience_level);
              return (
                <div
                  key={index}
                  onClick={() => {
                    enanbledNext(false);
                    enanbledPrev(false);
                    setSummery(item?.summary);
                  }}
                  className="group p-6 border-2 border-gray-200 rounded-xl cursor-pointer hover:border-black hover:shadow-lg transition-all duration-300 bg-gray-50 hover:bg-white"
                >
                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 bg-black group-hover:bg-gray-900 rounded-full flex items-center justify-center transition-colors duration-300">
                      <IconComponent className="w-5 h-5 text-white" strokeWidth={2} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="font-bold text-gray-900 group-hover:text-black transition-colors duration-300">
                          {item?.experience_level}
                        </h3>
                        <div className="w-2 h-2 bg-black rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                      </div>
                      <p className="text-gray-700 group-hover:text-gray-900 transition-colors duration-300 leading-relaxed">
                        {item?.summary}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default Summary;