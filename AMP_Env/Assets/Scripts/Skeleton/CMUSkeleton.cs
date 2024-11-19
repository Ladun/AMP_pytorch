using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;

namespace AMP
{
    public class CMUSkeleton : Skeleton
    {        

        const float LENGTH_SCALE = 0.5f;

        private ASFParser parser = new ASFParser();
        private Dictionary<string, Transform> boneTransforms = new Dictionary<string, Transform>();

        #region ASF Parse
        #endregion

        #region Control ans create Skeleton
        public void ResetSkeleton()
        {
            boneTransforms.Clear();

            if(transform.childCount > 0)
            {
                for(int i = 0; i < transform.childCount; i++)
                {
                    DestroyImmediate(transform.GetChild(i).gameObject); 
                }
            }
        }

        public override void CreateSkeleton()
        {
            ResetSkeleton();
            parser.ParseASF(skeletonFile);

            GameObject root = new GameObject("Root");
            root.transform.SetParent(transform);
            root.AddComponent<Rigidbody>();
            boneTransforms["root"] = root.transform;
            root.transform.localPosition = Vector3.zero;

            foreach (var entry in parser.bones)
            {
                string boneName = entry.Key;
                ASFParser.Bone bone = entry.Value;

                GameObject boneObject = CreateBoneObject(bone);
                boneObject.transform.SetParent(root.transform);
                boneTransforms[boneName] = boneObject.transform;
            }

            SetupHierarchy();
            SetBoneTransforms();
        }

        public override void SetAnimationData(MotionFrameData motionFrameData)
        {

            foreach (var entry in motionFrameData.JointData)
            {
                string jointName = entry.Key;
                List<float> data = entry.Value;
                if (boneTransforms.TryGetValue(jointName, out Transform bone))
                {
                    if (jointName == "root")
                    {
                        // CMU to Unity coordinate system
                        bone.localPosition = new Vector3(data[0], data[1], data[2]) * LENGTH_SCALE * parser.lengthScale;

                        // Apply root rotation
                        bone.localRotation = Quaternion.Euler(data[3], data[4], data[5]);
                    }
                    else
                    {
                        var boneData = parser.bones[jointName];
                        var dof = boneData.dof;
                        Vector3 euler = Vector3.zero;

                        for (int i = 0; i < dof.Count; i++)
                        {
                            float d = data[i];

                            if (dof[i] == "rx")
                                euler.x = d;
                            else if (dof[i] == "ry")
                                euler.y = d;
                            else if (dof[i] == "rz")
                                euler.z = d;
                        }

                        // Create rotation from Euler angles
                        Quaternion rotation = Quaternion.Euler(euler);

                        // Calculate the rotation to align the bone with its direction
                        Vector3 r_axis = Vector3.Cross(Vector3.forward, boneData.direction);
                        float angle = Vector3.SignedAngle(Vector3.forward, boneData.direction, r_axis);
                        Quaternion alignRotation = Quaternion.AngleAxis(angle, r_axis);

                        // Apply rotations in correct order: align rotation, then animation rotation, then initial rotation
                        bone.rotation = alignRotation * rotation;
                    }
                }
            }
        }
        

        private void SetupHierarchy()
        {
            foreach (var entry in parser.bones)
            {
                ASFParser.Bone bone = entry.Value;
                string childName = entry.Key;
                string parentName = bone.parentName;

                if (boneTransforms.TryGetValue(childName, out Transform child) &&
                    boneTransforms.TryGetValue(parentName, out Transform parent))
                {
                    child.SetParent(parent);
                }
                else
                {
                    Debug.LogWarning($"Could not set parent for {childName} to {parentName}");
                }
            }
        }

        private void SetBoneTransforms()
        {
            foreach (var entry in parser.bones)
            {
                string boneName = entry.Key;
                ASFParser.Bone boneData = entry.Value;
                float parentBoneLength = 0;
                Quaternion rotParentCurrent = Quaternion.identity;

                if(boneData.parentName != "root")
                {
                    if (parser.bones.TryGetValue(boneData.parentName, out ASFParser.Bone parentBoneData))
                    {
                        parentBoneLength = parentBoneData.length;
                        rotParentCurrent = Quaternion.Euler(boneData.axis) * Quaternion.Euler(-parentBoneData.axis);
                    }
                    else
                    {
                        Debug.LogWarning($"Could not find parent bone '{boneData.parentName}' for bone {boneName}");
                    }
                }

                if (boneTransforms.TryGetValue(boneName, out Transform bone) && 
                    boneTransforms.TryGetValue(boneData.parentName, out Transform parentBone))
                {
                    float ls = LENGTH_SCALE * parser.lengthScale;

                    bone.rotation = Quaternion.LookRotation(boneData.direction);
                    bone.position = parentBone.transform.position + parentBone.transform.forward * (parentBoneLength * 0.5f * ls);


                    // Create joint
                    var joint = bone.GetComponent<CharacterJoint>();
                    joint.connectedBody = parentBone.transform.GetComponent<Rigidbody>();

                    Rigidbody rigid = bone.GetComponent<Rigidbody>();
                    rigid.interpolation = RigidbodyInterpolation.Interpolate;
                    rigid.collisionDetectionMode = CollisionDetectionMode.Continuous;

                }
                else
                {
                    Debug.LogWarning($"Could not find transform for bone {boneName}");
                }
            }
        }

        private GameObject CreateBoneObject(ASFParser.Bone bone)
        {
            GameObject boneObject = new GameObject(bone.name);

            CharacterJoint joint = boneObject.AddComponent<CharacterJoint>();
            ConfigureJoint(joint, bone);

            CreateVisualRepresentation(boneObject, bone);

            return boneObject;
        }

        private void CreateVisualRepresentation(GameObject boneObject, ASFParser.Bone bone)
        {
            GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(visual.GetComponent<SphereCollider>());
            var col = visual.AddComponent<CapsuleCollider>();
            col.direction = 2;
            col.height = 0.9f;

            visual.transform.SetParent(boneObject.transform);
            float scaledLength = bone.length * parser.lengthScale * LENGTH_SCALE;
            visual.transform.localScale = new Vector3(0.1f, 0.1f, scaledLength / 2);
            visual.transform.localPosition = new Vector3(0, 0, scaledLength / 4);
        }

        private void ConfigureJoint(CharacterJoint joint, ASFParser.Bone bone)
        {
            // Convert from ASF coordinate system to Unity coordinate system
            joint.axis = new Vector3(bone.axis.x, bone.axis.y, bone.axis.z);
            joint.swingAxis = Vector3.Cross(joint.axis, Vector3.up);


            if (bone.limits != null && bone.limits.Length > 0)
            {
                ConfigureJointLimits(joint, bone.limits);
            }
        }

        private void ConfigureJointLimits(CharacterJoint joint, Vector2[] limits)
        {
            if (limits.Length > 0)
            {
                joint.lowTwistLimit = new SoftJointLimit { limit = -limits[0].x };
                joint.highTwistLimit = new SoftJointLimit { limit = limits[0].y };
            }
            if (limits.Length > 1)
            {
                joint.swing1Limit = new SoftJointLimit { limit = (Mathf.Abs(limits[1].x) + Mathf.Abs(limits[1].y)) / 2 };

            }
            if (limits.Length > 2)
            {
                joint.swing2Limit = new SoftJointLimit { limit = (Mathf.Abs(limits[2].x) + Mathf.Abs(limits[2].y)) / 2 };
                
            }
        }

        public override Observastion GetObs()
        {
            throw new NotImplementedException();
        }

        public override List<Transform> GetJoints()
        {
            throw new NotImplementedException();
        }
        #endregion


        private void OnDrawGizmosSelected()
        {
            foreach(var kv in parser.bones)
            {
                var boneData = kv.Value;

                Transform bone = boneTransforms[boneData.name];
                Vector3 dir = (Quaternion.Euler(boneData.axis) * Vector3.forward ) * 0.2f; 

                Debug.DrawLine(bone.position, bone.position + dir, Color.red);
            }
            
        }

        public override void RecordPrevState()
        {
            throw new NotImplementedException();
        }

        public override Transform GetBody(int id)
        {
            throw new NotImplementedException();
        }
    }
}