<template>
  <div class="login-container">
    <el-form ref="loginForm" :model="loginForm" :rules="loginRules" class="login-form" auto-complete="on" label-position="left">

      <div class="title-container">
        <h5 class="title">基于改进YOLOv7与无人机辅助的<br/>
          城市交通道路质量监测系统</h5>
      </div>

      <el-form-item prop="username">
        <span class="svg-container">
          <svg-icon icon-class="user" />
        </span>
        <el-input
          ref="username"
          v-model="loginForm.username"
          placeholder="Username"
          name="username"
          type="text"
          tabindex="1"
          auto-complete="on"
        />
      </el-form-item>

      <el-form-item prop="password">
        <span class="svg-container">
          <svg-icon icon-class="password" />
        </span>
        <el-input
          :key="passwordType.pwd1"
          ref="password"
          v-model="loginForm.password"
          :type="passwordType.pwd1"
          placeholder="Password"
          name="password"
          tabindex="2"
          auto-complete="on"
          @keyup.enter.native="handleLogin"
        />
        <span class="show-pwd" @click="showPwd(1)">
          <span v-if="formstatus=='forgetpwd'">旧密码</span>&nbsp;
          <svg-icon :icon-class="passwordType.pwd1 === 'password' ? 'eye' : 'eye-open'" />
        </span>
      </el-form-item>

      <el-form-item v-if="formstatus=='forgetpwd'" prop="newpassword">
        <span class="svg-container">
          <svg-icon icon-class="password" />
        </span>
        <el-input
          :key="passwordType.pwd2"
          ref="password"
          v-model="loginForm.newpassword"
          :type="passwordType.pwd2"
          placeholder="NewPassword"
          name="newpassword"
          tabindex="3"
          auto-complete="on"
          @keyup.enter.native="changePwd"
        />
        <span class="show-pwd" @click="showPwd(2)">
          <span v-if="formstatus=='forgetpwd'">新密码</span>&nbsp;
          <svg-icon :icon-class="passwordType.pwd2 === 'password' ? 'eye' : 'eye-open'" />
        </span>
      </el-form-item>

      <el-button :loading="loading" type="primary" style="width:100%;margin-bottom:30px;" @click.native.prevent="handleLogin">
        {{ formstatus=='login'?"登录":formstatus=='forgetpwd'?'确认修改':"注册" }}</el-button>

      <!--      :to="{path:`/forgetpwd?username=${loginForm.username}`-->
      <div class="tips">
        <a v-show="formstatus=='login'" @click="forgetPwd">忘记密码</a>
        <span v-show="formstatus=='login'">|</span>
        <a @click="changeStatus">{{ '去'+(formstatus=='login'?"注册":"登录") }}</a>
      </div>

    </el-form>
  </div>
</template>

<script>
import { validUsername } from '@/utils/validate'

const formstatus = { // 登录注册状态
  login: 'login',
  register: 'register',
  forgetpwd: 'forgetpwd'
}

export default {
  name: 'Login',
  data() {
    const validateUsername = (rule, value, callback) => {
      if (!validUsername(value)) {
        callback(new Error('Please enter the correct user name'))
      } else {
        callback()
      }
    }
    const validatePassword = (rule, value, callback) => {
      if (value.length < 6) {
        callback(new Error('The password can not be less than 6 digits'))
      } else {
        callback()
      }
    }
    return {
      loginForm: {
        username: 'admin',
        password: '111111',
        newpassword: ''
      },
      loginRules: {
        username: [{ required: true, trigger: 'blur', validator: validateUsername }],
        password: [{ required: true, trigger: 'blur', validator: validatePassword }],
        newpassword: [{ required: true, trigger: 'blur', validator: validatePassword }]
      },
      formstatus: formstatus.login,
      loading: false,
      passwordType: {
        pwd1: 'password',
        pwd2: 'password'
      },
      redirect: undefined
    }
  },
  watch: {
    $route: {
      handler: function(route) {
        this.redirect = route.query && route.query.redirect
      },
      immediate: true
    }
  },
  methods: {
    showPwd(index) {
      const name = 'pwd' + index
      if (this.passwordType[name] === 'password') {
        this.passwordType[name] = ''
      } else {
        this.passwordType[name] = 'password'
      }
      this.$nextTick(() => {
        this.$refs.password.focus()
      })
    },
    handleLogin() {
      this.$refs.loginForm.validate(valid => {
        if (valid) {
          this.loading = true
          if (this.formstatus === formstatus.login) { // 登录
            this.$store.dispatch('user/login', this.loginForm).then(() => {
              this.$router.push({ path: this.redirect || '/' })
              this.loading = false
              this.$message({
                message: '登录成功',
                type: 'success'
              })
            }).catch(() => {
              this.loading = false
            })
          } else if (this.formstatus === formstatus.register) { // 注册
            this.$store.dispatch('user/register', this.loginForm).then(() => {
              this.loading = false
              this.$message({
                message: '注册成功',
                type: 'success'
              })
              this.formstatus = formstatus.login
            }).catch(() => {
              this.loading = false
            })
          } else {
            this.$store.dispatch('user/resetpwd', this.loginForm).then(() => {
              this.loading = false
              this.$message({
                message: '修改成功',
                type: 'success'
              })
              this.formstatus = formstatus.login
            }).catch(() => {
              this.loading = false
            })
          }
        } else {
          this.$message({
            message: '账号或密码不合法',
            type: 'warning'
          })
          return false
        }
      })
    },
    changeStatus() {
      if (this.formstatus === formstatus.login) {
        this.formstatus = formstatus.register
      } else {
        this.formstatus = formstatus.login
      }
    },
    forgetPwd() {
      this.formstatus = formstatus.forgetpwd
    }
  }
}
</script>

<style lang="scss">
/* 修复input 背景不协调 和光标变色 */
/* Detail see https://github.com/PanJiaChen/vue-element-admin/pull/927 */

$bg:#283443;
$light_gray:#fff;
$cursor: #fff;

@supports (-webkit-mask: none) and (not (cater-color: $cursor)) {
  .login-container .el-input input {
    color: $cursor;
  }
}

/* reset element-ui css */
.login-container {
  .el-input {
    display: inline-block;
    height: 47px;
    width: 85%;

    input {
      background: transparent;
      border: 0px;
      -webkit-appearance: none;
      border-radius: 0px;
      padding: 12px 5px 12px 15px;
      color: $light_gray;
      height: 47px;
      caret-color: $cursor;

      &:-webkit-autofill {
        box-shadow: 0 0 0px 1000px $bg inset !important;
        -webkit-text-fill-color: $cursor !important;
      }
    }
  }

  .el-form-item {
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(0, 0, 0, 0.1);
    border-radius: 5px;
    color: #454545;
  }
}
</style>

<style lang="scss" scoped>
$bg:#2d3a4b;
$dark_gray:#889aa4;
$light_gray:#eee;

.login-container {
  min-height: 100%;
  width: 100%;
  //background-color: $bg;
  background-image: url("../../assets/404_images/bgc3.jpg");
  background-size: 100% 100%;
  overflow: hidden;

  .login-form {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%,-50%);
    border-radius: 30px;
    width: 520px;
    max-width: 100%;
    padding: 40px 50px;
    margin: 0 auto;
    overflow: hidden;
    background:linear-gradient(to bottom, #0296f3, #624e4e);
  }

  .tips {
    font-size: 14px;
    color: #fff;
    margin-bottom: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    span {
      margin: 0px 20px;
      color: red;
    }
    a:hover {
      color: #20a0ff;
    }
  }

  .svg-container {
    padding: 6px 5px 6px 15px;
    color: $dark_gray;
    vertical-align: middle;
    width: 30px;
    display: inline-block;
  }

  .title-container {
    position: relative;

    .title {
      font-size: 26px;
      color: $light_gray;
      margin: 0px auto 40px auto;
      text-align: center;
      font-weight: bold;
    }
  }

  .show-pwd {
    position: absolute;
    right: 10px;
    top: 7px;
    font-size: 16px;
    color: $dark_gray;
    cursor: pointer;
    user-select: none;
  }
}
</style>
