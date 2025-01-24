using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Runtime.InteropServices.ComTypes;
using UnityEngine.Rendering;

namespace AMP
{
    [CustomEditor(typeof(AnimationPlay))]
    public class AnimationPlayEditor : Editor
    {

        private Rect animBarRect;
        private int motionKey = 0;

        public override void OnInspectorGUI()
        {

            AnimationPlay p = (AnimationPlay)target;
            serializedObject.Update();

            DrawCustomScriptField();
            EditorGUIUtility.labelWidth = 100; // 라벨 너비를 80픽셀로 설정
            EditorGUILayout.PropertyField(serializedObject.FindProperty("motionDatabase"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("skeleton"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("ignoreRootPos"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("ignoreRootRot"));
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Animation Play Property", EditorStyles.boldLabel);

            if (p.motionDatabase)
            {
                if (p.motionDatabase.HasMotion)
                {
                    var keys = p.motionDatabase.GetMotionKeys();

                    int newKey = EditorGUILayout.Popup(motionKey, keys);
                    if (newKey != motionKey)
                    {
                        p.LoadData(keys[newKey]);
                    }

                    int currentFrame = EditorGUILayout.IntField("Current Frame", p.currentFrame);
                    UpdateFrame(currentFrame);
                    GUILayout.Space(5);

                    if (p.frameData != null && p.currentFrameData != null)
                    {
                        DrawAnimationRegion();
                    }

                }

                GUILayout.Space(35);
                if (GUILayout.Button("Update motion database", GUILayout.Width(200)))
                {
                    p.motionDatabase.LoadDataset();
                    p.LoadData(p.motionDatabase.GetMotionKeys()[0]);
                }
            }
            HandleInput();

            serializedObject.ApplyModifiedProperties();
        }

        void DrawAnimationRegion()
        {
            AnimationPlay p = (AnimationPlay)target;

            Rect rect = GUILayoutUtility.GetRect(EditorGUIUtility.currentViewWidth - 40, 40);
            EditorGUI.DrawRect(rect, new Color(0.3f, 0.3f, 0.3f, 1));
            animBarRect = rect;

            // Animation에서 5, 10 Frame마다 회색 바를 그림.
            Rect valueRect = new Rect(rect.x, rect.y, 1, rect.height);
            for (int i = 5; i < p.totalFrame; i += 5)
            {
                valueRect.width = (i % 10 == 0) ? 2 : 1;
                valueRect.x = rect.x + ((float)i / (p.totalFrame - 1)) * rect.width - valueRect.width / 2;
                EditorGUI.DrawRect(valueRect, new Color(0f, 0f, 0f, .2f));
            }
            valueRect.width = 3;
            // 현재 값에 해당하는 위치에 파란색 사각형을 그립니다.
            valueRect.x = rect.x + ((float)p.currentFrame / (p.totalFrame - 1)) * rect.width - valueRect.width / 2;
            EditorGUI.DrawRect(valueRect, new Color(0f, .7f, .3f, 1f));

            GUIStyle customStyle = new GUIStyle(EditorStyles.label);
            customStyle.alignment = TextAnchor.MiddleCenter;
            // 현재 값을 텍스트로 표시합니다.
            GUI.Label(rect, p.currentFrame.ToString(), customStyle);


            // 최소값과 최대값을 표시합니다.
            customStyle.alignment = TextAnchor.UpperLeft;
            GUI.Label(new Rect(rect.x, rect.y + rect.height, 30, 15), "0",
                      customStyle);
            customStyle.alignment = TextAnchor.UpperRight;
            GUI.Label(new Rect(rect.x + rect.width - 100, rect.y + rect.height, 100, 15), (p.totalFrame - 1).ToString(),
                      customStyle);

            const float motionDataContentWidth = 19;
            const float padding = 3f;
            rect = GUILayoutUtility.GetRect(EditorGUIUtility.currentViewWidth - 100,
                                            (motionDataContentWidth + padding) * p.currentFrameData.JointData.Count + padding);
            rect.width -= 200;
            rect.x += 100;
            rect.y += 20;
            EditorGUI.DrawRect(rect, new Color(0f, 0, 0, 0.3f));
            Vector2 motionDataRectPos = new Vector2(rect.x + padding, rect.y + padding);

            float contentNameSize = (rect.width - padding * 3) * 0.2f;
            float valueSize = (rect.width - padding * 3) * 0.8f;
            foreach (var kv in p.currentFrameData.JointData)
            {
                Rect nr = new Rect(motionDataRectPos, new Vector2(contentNameSize, motionDataContentWidth));
                EditorGUI.DrawRect(nr, new Color(1, 1, 1, 0.2f));
                GUI.Label(nr, kv.Key.ToString(), EditorStyles.boldLabel);

                Rect vr = new Rect(motionDataRectPos + new Vector2(contentNameSize + padding, 0), new Vector2(valueSize, motionDataContentWidth));
                EditorGUI.DrawRect(vr, new Color(1, 1, 1, 0.17f));
                for (int i = 0; i < kv.Value.Count; i++)
                {
                    GUI.Label(vr, (Mathf.Round(kv.Value[i] * 1000) / 1000.0f).ToString("F3"));
                    vr.x += 55;
                }

                motionDataRectPos.y += motionDataContentWidth + padding;
            }
        }


        void HandleInput()
        {
            AnimationPlay p = (AnimationPlay)target;
            if (p.frameData == null || p.currentFrameData == null)
                return;

            Event e = Event.current;

            if ((e.type == EventType.MouseDown || e.type == EventType.MouseDrag) && e.button == 0)
            {
                if (animBarRect.Contains(e.mousePosition))
                {
                    float percent = Mathf.InverseLerp(animBarRect.x, animBarRect.x + animBarRect.width, e.mousePosition.x);
                    UpdateFrame((int)(p.totalFrame * percent));
                }
            }
        }

        void UpdateFrame(int currentFrame)
        {
            AnimationPlay p = (AnimationPlay)target;

            if (currentFrame < 0)
                currentFrame = 0;
            if (currentFrame >= p.totalFrame)
                currentFrame = p.totalFrame - 1;
            if (p.currentFrame != currentFrame)
            {
                p.currentFrame = currentFrame;
                p.PlayAnimation();
            }
        }

        private void DrawCustomScriptField()
        {
            GUI.enabled = false;
            SerializedProperty scriptProperty = serializedObject.FindProperty("m_Script");
            EditorGUILayout.PropertyField(scriptProperty);
            GUI.enabled = true;
        }
    }
}

