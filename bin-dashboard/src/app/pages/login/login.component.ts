import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { userAuth } from '../../supabase.auth';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [ReactiveFormsModule], // <- Import ReactiveFormsModule for form handling
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  loginForm: FormGroup;

  constructor(private fb: FormBuilder) {
    this.loginForm = this.fb.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', Validators.required]
    });
  }

  get email() {
    return this.loginForm.get('email');
  }

  get password() {
    return this.loginForm.get('password');
  }

  async onSubmit() {
    if (this.loginForm.valid) {
      const { email, password } = this.loginForm.value;
      
      const { data, error } = await userAuth.auth.signInWithPassword({
        email,
        password
      })

      if(error) {
        console.log('Login error: ', error.message)
      }
      else {
        console.log('Signin successful: ', data)
        //redirect to user dashboard
      }

    } else {
      console.log('Form is invalid');
    }
  }
}
