<template>
  <div class="root">
    <el-card class="left">
      <div class="original_img">
        <el-divider content-position="left">文件上传 (支持多选)</el-divider>
        <el-upload
          ref="upload"
          :action="getBaseUrl()+'/uploadimg'"
          list-type="picture-card"
          multiple
          :limit="18"
          :on-preview="handlePictureCardPreview"
          :on-remove="handleRemove"
          :on-success="handleSuccess"
          :headers="{token:token}"
        >
          <i class="el-icon-plus" />
        </el-upload>
        <el-dialog v-if="dialogVisible" :visible.sync="dialogVisible">
          <img width="100%" :src="dialogImg.url" alt="" title="点击在新窗口预览" @click="openImg">
          <div><b>纬度：</b>{{ dialogImg.propties.Latitude }}</div>
          <div><b>经度：</b>{{ dialogImg.propties.Longitude }}</div>
          <div><b>裂缝地址：</b>{{ dialogImg.propties.Address[0] }}</div>
          <div><b>拍摄时间：</b>{{ handlerDate(dialogImg.propties.Time_taken) }}</div>
          <div><b>裂缝覆盖面积：</b>{{ 0.095 }}(平方米)</div>
          <div><b>危害程度：</b>中等</div>
        </el-dialog>
      </div>
      <el-divider content-position="left">模型检测结果</el-divider>

      <div v-show="imglist.length" class="new_img">
        <img v-for="(item,index) in imglist" :key="index" :src="item.url" title="预览" alt="预览" @click="handlePictureCardPreview(item)">
      </div>
    </el-card>
    <el-card class="contronal" :style="{height:screenHeight+'px'}">
      <div slot="header" class="clearfix">
        <span style="font-size: 20px;font-weight: 700">控制台</span>
        <button style="float: right; padding: 5px" class="choose_button">确认设置</button>
      </div>
      <el-card class="showdata">
        <!--        <div><b>检测速度：</b>{{ imglist.length?getRandom1():0 }} s</div>-->
        <div><b>目标数：</b>{{ imglist.length||0 }}</div>
        <div><b>检测速度：</b>{{ imglist.length?49.56:0 }} FPS</div>
      </el-card>
      <div class="showbtns">
        <div class="progress_msg">
          <div class="process_left">
            <div>NMS-IOU</div>
            <div>Confidence</div>
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
          <div class="choose_button el-icon-delete-solid" @click="clearModel">图片缓存清空</div>
          <div class="choose_button el-icon-download" @click="downLoadModel">检测图片下载</div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script>
import deviceMixin from '@/mixins/device'
import apiBaseUrl from '@/mixins/apiBaseUrl'
import { getToken } from '@/utils/auth'
import { downloadPic } from '@/utils/download'

export default {
  name: 'Photo',
  mixins: [deviceMixin, apiBaseUrl],
  data() {
    return {
      dialogImg: {},
      dialogVisible: false,
      imglist: [],
      token: '',
      progressState: [{
        id: 1,
        percentage: 30
      }, {
        id: 2,
        percentage: 50
      }]
    }
  },
  mounted() {
    this.token = getToken()
  },
  methods: {
    // getRandom1() {
    //   return ((Math.random() * 0.07) + 0.025).toFixed(2)
    // },
    // getRandom2() {
    //   return ((Math.random()) * 10 + 45).toFixed(2)
    // },
    handlerDate(date) {
      const data1 = date.split(' ')
      const data2 = data1[0].split(':')
      return data2[0] + '年' + data2[1] + '月' + data2[2] + '日 ' + data1[1]
    },
    handleRemove: function(file, fileList) {
    },
    handlePictureCardPreview(file) {
      this.dialogImg = file
      this.dialogVisible = true
    },
    handleSuccess(res, file, filelist) {
      res.url = '/dev-api' + res.url
      this.imglist.push(res)
    },
    openImg() {
      window.open(this.dialogImg.url)
    },
    clearModel() {
      this.$refs.upload.clearFiles()
      this.imglist = []
    },
    downLoadModel() {
      for (let i = 0; i < this.imglist.length; i++) {
        const urlname = this.imglist[i].url.split('/')
        const filename = urlname[urlname.length - 1]
        const url = 'http://127.0.0.1:3001' + this.imglist[i].url.split(process.env.VUE_APP_BASE_API).join('')
        downloadPic(url, filename)
      }
    },
    subProgress(item) {
      if (item.percentage <= 0) return
      item.percentage -= 10
    },
    addProgress(item) {
      if (item.percentage >= 100) return
      item.percentage += 10
    }
  }
}
</script>

<style scoped lang="scss">
.root {
  display: flex;
  .left {
    flex: 2;

    .original_img {

    }
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

    .new_img{
      img {
        width: 146px;
        height: 146px;
        margin: 0px 4px;
        &:hover {
          cursor: pointer;
          transform: scale(1.1);
          transition: 0.5s;
        }
      }
    }
}

</style>
