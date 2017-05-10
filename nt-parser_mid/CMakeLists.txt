PROJECT(cnn:nt-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


ADD_EXECUTABLE(nt-parser_mid nt-parser_mid.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser_mid cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(nt-parser-gen_mid nt-parser-gen_mid.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser-gen_mid cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(nt-parser-lm_mid nt-parser-lm_mid.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser-lm_mid cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(nt-parser_mid_nob nt-parser_mid_nob.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser_mid_nob cnn ${Boost_LIBRARIES})

