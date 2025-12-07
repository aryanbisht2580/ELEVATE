
import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Trash2, FolderOpen, Plus, Code, Layers, Rocket } from "lucide-react";
import { Input } from "@/components/ui/input";
import SimpeRichTextEditor from "@/components/custom/SimpeRichTextEditor";
import { useDispatch } from "react-redux";
import { addResumeData } from "@/features/resume/resumeFeatures";
import { toast } from "sonner";
import { useParams } from "react-router-dom";
import { LoaderCircle } from "lucide-react";
import { updateThisResume } from "@/Services/resumeAPI";

const formFields = {
  projectName: "",
  techStack: "",
  projectSummary: "",
};

function Project({ resumeInfo, setEnabledNext, setEnabledPrev }) {
  const [projectList, setProjectList] = useState(resumeInfo?.projects || []);
  const [loading, setLoading] = useState(false);
  const { resume_id } = useParams();
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(addResumeData({ ...resumeInfo, projects: projectList }));
  }, [projectList]);

  // Added to handle external updates from job description optimization
  useEffect(() => {
    if (resumeInfo?.projects && JSON.stringify(resumeInfo.projects) !== JSON.stringify(projectList)) {
      setProjectList(resumeInfo?.projects || []);
    }
  }, [resumeInfo?.projects]);

  const addProject = () => {
    setProjectList([...projectList, formFields]);
  };

  const removeProject = (index) => {
    const list = [...projectList];
    const newList = list.filter((item, i) => {
      if (i !== index) return true;
    });
    setProjectList(newList);
  };

  const handleChange = (e, index) => {
    setEnabledNext(false);
    setEnabledPrev(false);
    console.log("Type: ", typeof setEnabledPrev);
    const { name, value } = e.target;
    const list = [...projectList];
    const newListData = {
      ...list[index],
      [name]: value,
    };
    list[index] = newListData;
    setProjectList(list);
  };

  const handleRichTextEditor = (value, name, index) => {
    const list = [...projectList];
    const newListData = {
      ...list[index],
      [name]: value,
    };
    list[index] = newListData;
    setProjectList(list);
  };

  const onSave = () => {
    setLoading(true);
    const data = {
      data: {
        projects: projectList,
      },
    };
    if (resume_id) {
      console.log("Started Updating Project");
      updateThisResume(resume_id, data)
        .then((data) => {
          toast("Resume Updated", "success");
        })
        .catch((error) => {
          toast("Error updating resume", `${error.message}`);
        })
        .finally(() => {
          setEnabledNext(true);
          setEnabledPrev(true);
          setLoading(false);
        });
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
              <FolderOpen className="w-8 h-8 text-black" strokeWidth={2} />
            </div>
            <div className="text-white">
              <h2 className="text-2xl font-bold">Projects Portfolio</h2>
              <p className="text-gray-300 mt-1">Highlight your technical projects and innovations</p>
            </div>
          </div>
        </div>

        
        <div className="p-8 -mt-6 relative space-y-6">
          {projectList?.map((project, index) => (
            <div key={index} className="bg-white rounded-xl border-2 border-gray-100 shadow-sm overflow-hidden">
              
              <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-4 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center">
                      <Rocket className="w-5 h-5 text-white" strokeWidth={2} />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-gray-900">
                        Project {index + 1}
                      </h3>
                      <div className="w-12 h-0.5 bg-black rounded-full mt-1"></div>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    onClick={() => removeProject(index)}
                    className="border-2 border-red-200 text-red-600 hover:border-red-500 hover:bg-red-50 rounded-lg p-2 transition-all duration-200"
                  >
                    <Trash2 className="w-4 h-4" strokeWidth={2} />
                  </Button>
                </div>
              </div>

              
              <div className="p-6 space-y-6">
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <FolderOpen className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>Project Name</span>
                      <span className="text-red-500">*</span>
                    </label>
                    <div className="relative group">
                      <Input
                        type="text"
                        name="projectName"
                        value={project?.projectName}
                        onChange={(e) => handleChange(e, index)}
                        placeholder="e.g. E-commerce Platform"
                        className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black focus:ring-0 transition-all duration-200"
                      />
                      <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gray-200 rounded-full">
                        <div className="h-full bg-black rounded-full transition-all duration-300 scale-x-0 group-focus-within:scale-x-100 origin-left"></div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <Code className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>Technology Stack</span>
                    </label>
                    <div className="relative group">
                      <Input
                        type="text"
                        name="techStack"
                        value={project?.techStack}
                        placeholder="React, Node.js, Express, MongoDB"
                        onChange={(e) => handleChange(e, index)}
                        className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black focus:ring-0 transition-all duration-200"
                      />
                      <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gray-200 rounded-full">
                        <div className="h-full bg-black rounded-full transition-all duration-300 scale-x-0 group-focus-within:scale-x-100 origin-left"></div>
                      </div>
                    </div>
                  </div>
                </div>

                
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                    <Layers className="w-4 h-4 text-gray-500" strokeWidth={2} />
                    <span>Project Description & Key Features</span>
                  </label>
                  <div className="bg-gray-50 border-2 border-gray-200 rounded-lg p-4 focus-within:bg-white focus-within:border-black transition-all duration-200">
                    <SimpeRichTextEditor
                      index={index}
                      defaultValue={project?.projectSummary}
                      onRichTextEditorChange={(event) =>
                        handleRichTextEditor(event, "projectSummary", index)
                      }
                      resumeInfo={resumeInfo}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}

          
          {!projectList || projectList.length === 0 && (
            <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl p-12 text-center">
              <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                <FolderOpen className="w-8 h-8 text-gray-400" strokeWidth={2} />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Projects Added</h3>
              <p className="text-gray-500 mb-6">Start showcasing your technical projects and achievements</p>
              <Button
                onClick={addProject}
                className="bg-black text-white hover:bg-gray-800 rounded-lg px-6 py-3 font-medium transition-all duration-200"
              >
                <Plus className="w-4 h-4 mr-2" strokeWidth={2} />
                Add Your First Project
              </Button>
            </div>
          )}

          
          <div className="pt-6 border-t border-gray-100">
            <div className="flex justify-between items-center">
              <Button
                onClick={addProject}
                variant="outline"
                className="border-2 border-black text-black hover:bg-black hover:text-white transition-all duration-200 rounded-lg px-6 py-3 font-medium"
              >
                <Plus className="w-4 h-4 mr-2" strokeWidth={2} />
                Add {projectList?.length > 0 ? "More" : ""} Project
              </Button>
              
              <Button 
                onClick={onSave}
                disabled={loading}
                className="px-8 py-3 bg-gradient-to-r from-gray-900 to-black text-white hover:from-black hover:to-gray-900 disabled:from-gray-400 disabled:to-gray-500 rounded-lg font-medium transition-all duration-200 shadow-lg hover:shadow-xl min-w-[120px]"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <LoaderCircle className="w-4 h-4 animate-spin" />
                    <span>Saving...</span>
                  </div>
                ) : (
                  "Save Projects"
                )}
              </Button>
            </div>
          </div>

          
        </div>

        <div className="h-1 bg-gradient-to-r from-gray-900 to-black"></div>
      </div>
    </div>
  );
}

export default Project;