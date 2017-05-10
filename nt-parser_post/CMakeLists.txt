PROJECT(cnn:nt-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


ADD_EXECUTABLE(nt-parser_post nt-parser_post.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser_post cnn ${Boost_LIBRARIES})

