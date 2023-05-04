<template>
  <div class="video_root">
    <!-- 左侧原视频   -->
    <el-card v-if="videoUrl==''" class="original_video">
      <el-skeleton :loading="loading" animated>
        <template slot="template">
          <el-skeleton-item
            variant="image"
            :style="{width: '100%', height:(screenHeight-50)+'px'}"
          />
        </template>
        <template>
          <el-upload
            ref="upload"
            class="avatar-uploader"
            :action="getBaseUrl()+'/uploadvideo'"
            :show-file-list="false"
            :on-success="handleSuccess"
            :before-upload="handleBeforeUpload"
            :headers="{token:token}"
          >
            <i class="el-icon-plus avatar-uploader-icon" />
          </el-upload>
        </template>
      </el-skeleton>
    </el-card>
    <!--    左侧模型-->
    <el-card v-else class="new_video">
      <video :src="videoUrl" autoplay controls="controls" crossOrigin="Anonymous">
        您的浏览器不支持 video 标签。
      </video>
    </el-card>
    <!--    控制台-->
    <el-card class="contronal" :style="{height:screenHeight+'px'}">
      <div slot="header" class="clearfix">
        <span style="font-size: 20px;font-weight: 700">控制台</span>
        <button style="float: right; padding: 5px" class="choose_button">确认设置</button>
      </div>
      <el-card class="showdata">
        <div><b>检测用时：</b>{{ videoUrl?15.0:0 }} s</div>
        <div><b>目标数：</b>{{ videoUrl?89:0 }}</div>
        <div><b>检测速度：</b>{{ videoUrl?41:0 }} FPS</div>
      </el-card>
      <div class="showbtns">
        <div class="progress_msg">
          <div class="process_left">
            <div>NMS-IOU</div>
            <div>Confidence</div>
<!--            <div>帧间延迟</div>-->
          </div>
          <div class="process_center">
            <el-progress v-for="(item,index) in progressState" :key="index" :text-inside="true" :stroke-width="20" :percentage="item.percentage" status="exception" />
          </div>
          <div class="process_right">
            <div v-for="(item,index) in progressState" :key="index" class="buttons">
              <span @click="subProgress(item)">-</span>
              <span @click="addProgress(item)">+</span>
            </div>
          </div>
        </div>
        <div class="other_choose">
          <div class="choose_button el-icon-delete-solid" @click="clearVideo">视频缓存清空</div>
          <div class="choose_button el-icon-download" @click="videoUrl&&downVideo(videoUrl)">检测视频下载</div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script>
import deviceMixin from '@/mixins/device'
import apiBaseUrl from '@/mixins/apiBaseUrl'
import { getToken } from '@/utils/auth'
import { downVideo } from '@/utils/download'
export default {
  name: 'Video',
  mixins: [deviceMixin, apiBaseUrl],
  ischecking: false, // 正在上传检测
  data() {
    return {
      dialogImageUrl: '',
      dialogVisible: false,
      videoUrl: '',
      loading: false,
      progressState: [{
        id: 1,
        percentage: 30
      }, {
        id: 2,
        percentage: 50
      }],
      token: ''
    }
  },
  mounted() {
    this.token = getToken()
  },
  methods: {
    // getRandom1() {
    //   return Math.floor((Math.random() * 10) + 11)
    // },
    // getRandom2() {
    //   return Math.floor((Math.random() * 2) + 40)
    // },
    clearVideo() {
      this.$refs.upload.clearFiles()
      this.imglist = []
    },
    handleBeforeUpload(file) {
      const { type } = file
      if (/video/.test(type)) {
        this.$message({
          message: '视频检测中...请稍候',
          type: 'success'
        })
        this.loading = true
        return true
      } else {
        this.$message({
          message: '只能上传视频',
          type: 'warning'
        })
        return false
      }
    },
    handlePictureCardPreview(file) {
      this.dialogImageUrl = file.url
      this.dialogVisible = true
    },
    handleSuccess(res, file) {
      this.videoUrl = process.env.VUE_APP_BASE_API + res.url
      this.$message({
        message: '视频检测成功',
        type: 'success'
      })
      this.loading = false
    },
    subProgress(item) {
      if (item.percentage <= 0) return
      item.percentage -= 10
    },
    addProgress(item) {
      if (item.percentage >= 100) return
      item.percentage += 10
    },
    downVideo(url) {
      downVideo(url)
    }
  }
}
</script>

<style scoped lang="scss">
.avatar-uploader-icon {
  border: 1px dashed #d9d9d9;
  font-size: 28px;
  color: #8c939d;
  width: 178px;
  height: 178px;
  line-height: 178px;
  text-align: center;
}
 .avatar-uploader-icon:hover {
  border-color: #409EFF;
}

 .video_root{
   display: flex;
   width: 100%;
   .original_video{
     flex: 1;
   }
   .new_video{
     flex: 1;
   }
   .contronal{
     width: 400px;
     height: 100%;
     overflow: auto;
     margin-left: 10px;
     .showdata{
       border-radius: 10px;
       margin-bottom: 20px;
       background-color: #8ee8f8;
       box-shadow: 1px 5px 5px 2px #20a0ff;
       div {
         width: 100%;
         margin: 6px 0;
       }
     }
     .showbtns{
       .progress_msg{
         width: 100%;
         display: flex;
         .process_left{
           width: 80px;
           font-size: 14px;
           div {
             margin: 6px 0;
             line-height: 20px;
           }
         }
         .process_center{
           flex: 1;
         }
         .process_right{
           width: 84px;
           .buttons{
             box-sizing: border-box;
             line-height: 20px;
             margin: 6px 0;
             span {
               display: inline-block;
               margin: 0 5px;
               text-align: center;
               width: 30px;
               cursor: pointer;
               background-color: #0173e7;
               border-radius: 5px;
               color: #fff;
               &:hover {
                 background-color: #1687ff;
               }
             }
           }
         }
       }

       .other_choose{
         margin: 10px 0px;
         display: flex;
         flex-wrap: wrap;
         .choose_button{
           margin:20px 30px 0px 0px;
         }
       }

       .el-progress{
         margin: 6px;
       }
     }
   }
 }
 video{
   width: 100%;
 }
</style>
