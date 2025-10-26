/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.visioncamerafacedetector.recognizer;

public interface SimilarityClassifier {


    /** An immutable result returned by a Classifier describing what was recognized. */
    class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        public final String userId;
        /** Display name for the recognition. */
        public final String faceName;


        public final Float distance;
        public Object extra;

        public String uri;

        public Recognition(
                final String userId, final String faceName, final Float distance, final String uri) {
            this.userId = userId;
            this.faceName = faceName;
            this.distance = distance;
            this.extra = null;
            this.uri = uri;
        }

        public void setExtra(Object extra) {
            this.extra = extra;
        }
        public Object getExtra() {
            return this.extra;
        }
    }
}