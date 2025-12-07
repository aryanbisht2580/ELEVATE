import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { LoaderCircle, Trash2, Briefcase, Plus, Building, MapPin, Calendar, User } from "lucide-react";
import RichTextEditor from "@/components/custom/RichTextEditor";
import React, { useEffect } from "react";
import { useDispatch } from "react-redux";
import { addResumeData } from "@/features/resume/resumeFeatures";
import { useParams } from "react-router-dom";
import { updateThisResume } from "@/Services/resumeAPI";
import { toast } from "sonner";

const formFields = {
  title: "",
  companyName: "",
  city: "",
  state: "",
  startDate: "",
  endDate: "",
  currentlyWorking: "",
  workSummary: "",
};

function Experience({ resumeInfo, enanbledNext, enanbledPrev }) {
  const [experienceList, setExperienceList] = React.useState(resumeInfo?.experience || []);
  const [loading, setLoading] = React.useState(false);
  const { resume_id } = useParams();
  const dispatch = useDispatch();

  useEffect(() => {
    try {
      dispatch(addResumeData({ ...resumeInfo, experience: experienceList }));
    } catch (error) {
      console.log("error in experience context update", error.message);
    }
  }, [experienceList]);

  // Added to handle external updates from job description optimization
  useEffect(() => {
    if (resumeInfo?.experience && JSON.stringify(resumeInfo.experience) !== JSON.stringify(experienceList)) {
      setExperienceList(resumeInfo?.experience || []);
    }
  }, [resumeInfo?.experience]);

  const addExperience = () => {
    if (!experienceList) {
      setExperienceList([formFields]);
      return;
    }
    setExperienceList([...experienceList, formFields]);
  };

  const removeExperience = (index) => {
    const list = [...experienceList];
    const newList = list.filter((_, i) => i !== index);
    setExperienceList(newList);
  };

  const handleChange = (e, index) => {
    enanbledNext(false);
    enanbledPrev(false);
    const { name, value } = e.target;
    const list = [...experienceList];
    list[index] = { ...list[index], [name]: value };
    setExperienceList(list);
  };

  const handleRichTextEditor = (value, name, index) => {
    const list = [...experienceList];
    list[index] = { ...list[index], [name]: value };
    setExperienceList(list);
  };

  const onSave = () => {
    setLoading(true);
    const data = {
      data: {
        experience: experienceList,
      },
    };
    if (resume_id) {
      updateThisResume(resume_id, data)
        .then(() => toast("Resume Updated", "success"))
        .catch((error) => toast("Error updating resume", `${error.message}`))
        .finally(() => {
          enanbledNext(true);
          enanbledPrev(true);
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
              <Briefcase className="w-8 h-8 text-black" strokeWidth={2} />
            </div>
            <div className="text-white">
              <h2 className="text-2xl font-bold">Professional Experience</h2>
              <p className="text-gray-300 mt-1">Showcase your career journey and achievements</p>
            </div>
          </div>
        </div>

        <div className="p-8 -mt-6 relative space-y-6">
          {experienceList?.map((experience, index) => (
            <div key={index} className="bg-white rounded-xl border-2 border-gray-100 shadow-sm overflow-hidden">
              <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-4 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center">
                      <User className="w-5 h-5 text-white" strokeWidth={2} />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-gray-900">Experience {index + 1}</h3>
                      <div className="w-12 h-0.5 bg-black rounded-full mt-1"></div>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    onClick={() => removeExperience(index)}
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
                      <Briefcase className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>Position Title</span>
                    </label>
                    <Input
                      type="text"
                      name="title"
                      value={experience?.title}
                      onChange={(e) => handleChange(e, index)}
                      placeholder="e.g. Senior Software Engineer"
                      className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <Building className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>Company Name</span>
                    </label>
                    <Input
                      type="text"
                      name="companyName"
                      value={experience?.companyName}
                      onChange={(e) => handleChange(e, index)}
                      placeholder="e.g. Google Inc."
                      className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <MapPin className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>City</span>
                    </label>
                    <Input
                      type="text"
                      name="city"
                      value={experience?.city}
                      onChange={(e) => handleChange(e, index)}
                      placeholder="e.g. San Francisco"
                      className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <MapPin className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>State</span>
                    </label>
                    <Input
                      type="text"
                      name="state"
                      value={experience?.state}
                      onChange={(e) => handleChange(e, index)}
                      placeholder="e.g. California"
                      className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>Start Date</span>
                    </label>
                    <Input
                      type="text"
                      name="startDate"
                      value={experience?.startDate}
                      onChange={(e) => handleChange(e, index)}
                      placeholder="e.g. Jan 2022"
                      className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-gray-500" strokeWidth={2} />
                      <span>End Date</span>
                    </label>
                    <Input
                      type="text"
                      name="endDate"
                      value={experience?.endDate}
                      onChange={(e) => handleChange(e, index)}
                      placeholder="e.g. Dec 2023 or Present"
                      className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-lg text-gray-900 placeholder-gray-500 focus:bg-white focus:border-black"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700 flex items-center space-x-2">
                    <Briefcase className="w-4 h-4 text-gray-500" strokeWidth={2} />
                    <span>Job Description & Achievements</span>
                  </label>
                  <div className="bg-gray-50 border-2 border-gray-200 rounded-lg p-4">
                    <RichTextEditor
                      index={index}
                      defaultValue={experience?.workSummary}
                      onRichTextEditorChange={(value) =>
                        handleRichTextEditor(value, "workSummary", index)
                      }
                      resumeInfo={resumeInfo}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}

          {!experienceList || experienceList.length === 0 ? (
            <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl p-12 text-center">
              <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                <Briefcase className="w-8 h-8 text-gray-400" strokeWidth={2} />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Experience Added</h3>
              <p className="text-gray-500 mb-6">Start building your professional experience section</p>
              <Button
                onClick={addExperience}
                className="bg-black text-white hover:bg-gray-800 rounded-lg px-6 py-3 font-medium"
              >
                <Plus className="w-4 h-4 mr-2" strokeWidth={2} />
                Add Your First Experience
              </Button>
            </div>
          ) : null}

          <div className="pt-6 border-t border-gray-100">
            <div className="flex justify-between items-center">
              <Button
                onClick={addExperience}
                variant="outline"
                className="border-2 border-black text-black hover:bg-black hover:text-white rounded-lg px-6 py-3 font-medium"
              >
                <Plus className="w-4 h-4 mr-2" strokeWidth={2} />
                Add {experienceList?.length > 0 ? "More" : ""} Experience
              </Button>
              <Button
                onClick={onSave}
                disabled={loading}
                className="px-8 py-3 bg-gradient-to-r from-gray-900 to-black text-white hover:from-black hover:to-gray-900 disabled:from-gray-400 disabled:to-gray-500 rounded-lg font-medium shadow-lg hover:shadow-xl min-w-[120px]"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <LoaderCircle className="w-4 h-4 animate-spin" />
                    <span>Saving...</span>
                  </div>
                ) : (
                  "Save Experience"
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

export default Experience;
