<template>
  <div class="app-container">
    <div class="header">
      <el-input
        v-model="username"
        placeholder="账号"
        clearable
      />
      <el-button type="primary" icon="el-icon-search" @click="searchTable">搜索</el-button>
    </div>
    <el-divider />
    <el-table
      v-loading="listLoading"
      :data="pagenationData.currentData"
      element-loading-text="Loading"
      :max-height="screenHeight"
      border
      fit
      highlight-current-row
    >
      <el-table-column align="center" label="ID" width="95">
        <template slot-scope="scope">
          {{ scope.$index }}
        </template>
      </el-table-column>
      <el-table-column label="用户名">
        <template slot-scope="scope">
          {{ scope.row.user_id }}
        </template>
      </el-table-column>
      <el-table-column label="检测类别" align="center">
        <template slot-scope="scope">
          {{ scope.row.check_type=='img'?"图片":"视频" }}
        </template>
      </el-table-column>
      <el-table-column label="原始图片/视频" align="center">
        <template slot-scope="scope">
          <img v-if="scope.row.check_type=='img'" style="width: 150px;height: auto" :src="'/dev-api'+scope.row.srcurl" title="点击预览" @click="handlePictureCardPreview('dev-api'+scope.row.srcurl,'img')">
          <video v-else style="width: 150px;height: auto" :src="'/dev-api'+scope.row.srcurl" crossOrigin="Anonymous" @click="handlePictureCardPreview('dev-api'+scope.row.srcurl,'video')" />
        </template>
      </el-table-column>
      <el-table-column label="检测结果" align="center">
        <template slot-scope="scope">
          <img v-if="scope.row.check_type=='img'" style="width: 150px;height: auto" :src="'/dev-api'+scope.row.url" title="点击预览" @click="handlePictureCardPreview('dev-api'+scope.row.url,'img')">
          <video v-else style="width: 150px;height: auto" :src="'/dev-api'+scope.row.url" crossOrigin="Anonymous" @click="handlePictureCardPreview('dev-api'+scope.row.url,'video')" />
        </template>
      </el-table-column>
      <el-table-column label="检测时间" align="center">
        <template slot-scope="scope">
          {{ handlerTime(parseInt(scope.row.check_time)) }}
        </template>
      </el-table-column>
      <el-table-column class-name="status-col" label="详细信息" align="center">
        <template slot-scope="scope">
          <div v-html="scope.row.detail"></div>
        </template>
      </el-table-column>
      <el-table-column align="center" prop="created_at" label="操作" width="180">
        <template slot-scope="scope">
          <el-button
            size="mini"
            @click="handleDown(scope.$index, scope.row)"
          >下载</el-button>
          <el-button
            size="mini"
            type="danger"
            @click="handleDelete(scope.$index, scope.row)"
          >删除</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-pagination
      hide-on-single-page
      background
      layout="prev, pager, next"
      :total="4"
    />
    <el-dialog v-if="dialogVisible" :visible.sync="dialogVisible">
      <img v-if="dialogImg" width="100%" :src="dialogImg" alt="" title="点击在新窗口预览" @click="openImg">
      <video v-if="dialogVideo" style="width: 100%" :src="dialogVideo" crossOrigin="Anonymous" controls />
      <!--      <img width="100%" :src="dialogImg.url" alt="" title="点击在新窗口预览" @click="openImg">-->
      <!--      <div><b>纬度：</b>{{ dialogImg.propties.Latitude }}</div>-->
      <!--      <div><b>经度：</b>{{ dialogImg.propties.Longitude }}</div>-->
      <!--      <div><b>详细地址：</b>{{ dialogImg.propties.Address[0] }}</div>-->
      <!--      <div><b>拍摄时间：</b>{{ dialogImg.propties.Time_taken }}</div>-->
    </el-dialog>
  </div>
</template>

<script>
import { getList } from '@/api/checklist'
import { getScreenHeight } from '@/utils/deviceMsg'
import { downloadPic, downVideo } from '@/utils/download'

export default {
  filters: {
    statusFilter(status) {
      const statusMap = {
        published: 'success',
        draft: 'gray',
        deleted: 'danger'
      }
      return statusMap[status]
    }
  },
  data() {
    return {
      username: '',
      drawer: false,
      list: null,
      listcopy: null, // 备份数据，方便数据还原
      listLoading: true,
      dialogVisible: false,
      dialogImg: '',
      dialogVideo: '',
      formLabelAlign: {
        nickname: '',
        username: '',
        password: '',
        phone: ''
      },
      pagenationData: {
        totol: 0,
        currentIndex: 1,
        currentData: []
      }
    }
  },
  computed: {
    screenHeight() {
      return getScreenHeight() - 220
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    fetchData() {
      this.listLoading = true
      getList().then(response => {
        this.pagenationData.currentData = response.data.items
        this.pagenationData.currentData = this.pagenationData.currentData.map(el => {
          el.detail = '经度：' + this.getRandomx() + '<br/>纬度：' + this.getRandomy()
          return el
        })
        console.log(this.pagenationData.currentData)
        this.listcopy = this.list
        this.listLoading = false
      }).catch(() => {
        console.log('获取数据错误')
      })
    },
    getRandomx() {
      return (Math.random() * 3 + 111).toFixed(2)
    },
    getRandomy() {
      return (Math.random() * 1.5 + 27).toFixed(2)
    },
    searchTable() {
      if (!this.username) return
      this.list = this.list.filter(el => {
        return el.nickname === this.username
      })
      console.log(this.list)
    },
    handleClose(done) {
      this.$confirm('确认关闭？')
        .then(_ => {
          this.drawer = false
          done()
        })
        .catch(_ => {})
    },
    handleDown(index, row) {
      console.log(row)
      if (row.check_type === 'img') {
        const urls = row.url.split('/')
        downloadPic('http://127.0.0.1:3001' + row.url, parseInt(Math.random() * 1000) + urls[urls.length - 1])
      } else {
        downVideo('/dev-api' + row.url)
      }
    },
    handleDelete(index, row) {
      this.pagenationData.currentData.splice(index,1)
      this.$message({
        type:"success",
        message:"删除成功"
      })
      console.log(index, row)
    },
    handlePictureCardPreview(file, type) {
      if (type === 'img') {
        this.dialogImg = file
        this.dialogVideo = ''
      } else {
        this.dialogImg = ''
        this.dialogVideo = file
      }
      this.dialogVisible = true
    },
    openImg() {
      window.open(this.dialogImg)
    },
    handlerTime(time) {
      return String(new Date(time)).slice(10)
    }
  }
}
</script>

<style lang="scss">
.header {
  display: flex;
  .el-input{
    margin-right: 20px;
  }
}

.el-drawer{
  padding: 0px 20px;
}

.el-pagination{
  margin: 10px auto 0;
  float: right;
}
</style>
